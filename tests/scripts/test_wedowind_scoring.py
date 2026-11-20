from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.wedowind_wake_model_2026.scoring import (
    LEFT_LOS_COL,
    LR_NORMALISER_PCT,
    MIN_T03_POWER_KW,
    POWER_NORMALISER,
    RIGHT_LOS_COL,
    SCORING_PERIODS,
    SUBMISSION_COLS,
    T03_NORMAL_COL,
    T03_POWER_COL,
    T03_WD_COL,
    T03_WS_COL,
    T07_NORMAL_COL,
    T07_POWER_COL,
    TEMPLATE_PATH,
    TIMESTAMP_COL,
    TRUTH_COLS,
    VALUE_COLS,
    WD_RANGE_DEG,
    WS_RANGE_MS,
    read_submission,
    score_submission,
    scored_minutes,
    submission_template,
    submission_timestamps,
    validate_submission,
    write_submission_template,
)


def _truth() -> pd.DataFrame:
    """Every minute scored: in window, both turbines normal, LiDAR present, L-R of +20 % (L 11, R 9)."""
    index = submission_timestamps()
    return pd.DataFrame(
        {
            T03_WD_COL: 214.0,
            T03_WS_COL: 9.0,
            T03_POWER_COL: 1000.0,
            T03_NORMAL_COL: True,
            T07_NORMAL_COL: True,
            T07_POWER_COL: 700.0,
            LEFT_LOS_COL: 11.0,
            RIGHT_LOS_COL: 9.0,
        },
        index=index,
    )[list(TRUTH_COLS)]


def _submission_csv_df(values: pd.DataFrame) -> pd.DataFrame:
    return values.reset_index(names=TIMESTAMP_COL).assign(
        **{TIMESTAMP_COL: lambda df: df[TIMESTAMP_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")}
    )


def _perfect_submission() -> pd.DataFrame:
    return _submission_csv_df(_truth()[list(VALUE_COLS)])


def test_template_timestamps_are_every_minute_start_of_every_period_end_exclusive() -> None:
    """2,820 minute starts; each period starts on its start and stops a minute before its end."""
    timestamps = submission_timestamps()

    assert len(timestamps) == 2820
    assert timestamps.is_unique
    assert timestamps.is_monotonic_increasing
    assert str(timestamps.tz) == "UTC"
    assert (timestamps.second == 0).all()
    for start, end_excl in SCORING_PERIODS:
        assert start in timestamps
        assert end_excl not in timestamps
        assert end_excl - pd.Timedelta(minutes=1) in timestamps


def test_committed_template_matches_submission_timestamps(tmp_path: Path) -> None:
    """The committed blank template is what ``write_submission_template`` writes, so it can't drift."""
    regenerated = tmp_path / "template.csv"
    write_submission_template(regenerated)

    assert TEMPLATE_PATH.read_bytes() == regenerated.read_bytes()
    template = pd.read_csv(TEMPLATE_PATH)
    assert list(template.columns) == list(SUBMISSION_COLS)
    assert template.loc[0, TIMESTAMP_COL] == "2026-09-01T02:00:00Z"
    assert template[list(VALUE_COLS)].isna().all().all()
    assert submission_template().shape == (2820, 3)


def test_read_submission_round_trips_a_csv(tmp_path: Path) -> None:
    """Rows may come in any order; the values come back in template order."""
    path = tmp_path / "submission.csv"
    _perfect_submission().sample(frac=1, random_state=0).to_csv(path, index=False)

    values = read_submission(path)

    assert values.index.equals(submission_timestamps())
    pd.testing.assert_frame_equal(values, _truth()[list(VALUE_COLS)], check_names=False)


def test_rejects_missing_duplicated_or_extra_timestamps() -> None:
    """The submission must have exactly the template timestamps, once each."""
    sub = _perfect_submission()
    with pytest.raises(ValueError, match="missing timestamps"):
        validate_submission(sub.drop(index=5))
    with pytest.raises(ValueError, match="duplicated"):
        validate_submission(pd.concat([sub, sub.iloc[[5]]]))
    extra = sub.iloc[[0]].assign(**{TIMESTAMP_COL: "2026-09-01T11:00:00Z"})
    with pytest.raises(ValueError, match="1 extra"):
        validate_submission(pd.concat([sub, extra]))


@pytest.mark.parametrize("col", SUBMISSION_COLS)
def test_rejects_a_missing_column(col: str) -> None:
    """Each submission column, the timestamp included, is required."""
    with pytest.raises(ValueError, match="missing columns"):
        validate_submission(_perfect_submission().drop(columns=col))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "fast"])
@pytest.mark.parametrize("col", VALUE_COLS)
def test_rejects_non_finite_values(col: str, bad: object) -> None:
    """Every row must be predicted, even minutes that won't be scored."""
    sub = _perfect_submission().astype({col: object})
    sub.loc[7, col] = bad
    with pytest.raises(ValueError, match="non-finite"):
        validate_submission(sub)


@pytest.mark.parametrize("col", [LEFT_LOS_COL, RIGHT_LOS_COL])
def test_rejects_a_negative_los_speed(col: str) -> None:
    """The ZXTM reports LOS speed magnitudes, so a negative one is not a valid prediction."""
    sub = _perfect_submission()
    sub.loc[3, col] = -0.1
    with pytest.raises(ValueError, match="negative LOS speed"):
        validate_submission(sub)


def test_zero_los_speeds_are_accepted_and_scored_without_dividing_by_zero() -> None:
    """Calm is physically possible: both LOS zero gives L-R 0, scored against the measured +20 %."""
    sub = _perfect_submission()
    sub[[LEFT_LOS_COL, RIGHT_LOS_COL]] = 0.0

    result = score_submission(validate_submission(sub), _truth())

    assert result["lr_rmse_pct_points"] == pytest.approx(20.0)
    assert np.isfinite(result["score"])


def test_measured_zero_los_speeds_do_not_divide_by_zero() -> None:
    """The truth gets the same protection: both measured LOS zero gives a measured L-R of 0."""
    truth = _truth()
    truth[[LEFT_LOS_COL, RIGHT_LOS_COL]] = 0.0

    result = score_submission(validate_submission(_perfect_submission()), truth)

    assert result["lr_rmse_pct_points"] == pytest.approx(20.0)


def _one_minute_truth(**overrides: object) -> pd.DataFrame:
    return _truth().iloc[[0]].assign(**overrides)


@pytest.mark.parametrize(
    ("col", "value", "scored"),
    [
        (T03_WD_COL, WD_RANGE_DEG[0], True),
        (T03_WD_COL, WD_RANGE_DEG[0] - 1e-6, False),
        (T03_WD_COL, WD_RANGE_DEG[1] - 1e-6, True),
        (T03_WD_COL, WD_RANGE_DEG[1], False),
        (T03_WD_COL, np.nan, False),
        (T03_WS_COL, WS_RANGE_MS[0], True),
        (T03_WS_COL, WS_RANGE_MS[0] - 1e-6, False),
        (T03_WS_COL, WS_RANGE_MS[1] - 1e-6, True),
        (T03_WS_COL, WS_RANGE_MS[1], False),
        (T03_WS_COL, np.nan, False),
        (T03_POWER_COL, MIN_T03_POWER_KW, False),
        (T03_POWER_COL, MIN_T03_POWER_KW + 1e-6, True),
        (T03_NORMAL_COL, False, False),
        (T03_NORMAL_COL, pd.NA, False),
        (T07_NORMAL_COL, False, False),
        (T07_NORMAL_COL, pd.NA, False),
    ],
)
def test_scored_minutes_window_edges(col: str, value: object, *, scored: bool) -> None:
    """Direction and speed windows are [lo, hi); T03 power must exceed the minimum; missing never counts."""
    truth = _one_minute_truth()
    truth[col] = pd.Series([value], index=truth.index, dtype="boolean" if "normal" in col else float)

    mask = scored_minutes(truth)

    assert mask["power"].iloc[0] == scored
    assert mask["lidar"].iloc[0] == scored


@pytest.mark.parametrize("col", [LEFT_LOS_COL, RIGHT_LOS_COL])
def test_lidar_term_needs_both_measured_los(col: str) -> None:
    """A minute missing either measured LOS still counts for power but not for L-R."""
    mask = scored_minutes(_one_minute_truth(**{col: np.nan}))

    assert mask["power"].iloc[0]
    assert not mask["lidar"].iloc[0]


def test_perfect_submission_scores_zero() -> None:
    """Submitting the truth scores 0 with every minute scored."""
    result = score_submission(validate_submission(_perfect_submission()), _truth())

    assert result == {
        "score": 0.0,
        "power_rmse": 0.0,
        "lr_rmse_pct_points": 0.0,
        "power_minutes": 2820,
        "lidar_minutes": 2820,
    }


def _l_r_values(lr_pct: float, *, mean: float = 10.0) -> tuple[float, float]:
    """Return (L, R) with mean ``mean`` and 100 (L - R) / mean = ``lr_pct``."""
    half_diff = lr_pct * mean / 200
    return mean + half_diff, mean - half_diff


def test_no_steering_sized_errors_score_one() -> None:
    """A power error of the power normaliser and an L-R error of the L-R normaliser every minute score 1."""
    truth = _truth()
    values = truth[list(VALUE_COLS)].copy()
    values[T07_POWER_COL] += POWER_NORMALISER * truth[T03_POWER_COL]
    values[LEFT_LOS_COL], values[RIGHT_LOS_COL] = _l_r_values(20.0 - LR_NORMALISER_PCT)

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["power_rmse"] == pytest.approx(POWER_NORMALISER)
    assert result["lr_rmse_pct_points"] == pytest.approx(LR_NORMALISER_PCT)
    assert result["score"] == pytest.approx(1.0)


def test_terms_are_weighted_half_each_and_pooled_over_scored_minutes_only() -> None:
    """Each normalised term contributes half; unscored minutes, however wrong, don't count."""
    truth = _truth()
    unscored = truth.index[::2]
    truth.loc[unscored, T03_WS_COL] = 3.0
    truth.loc[truth.index[1::4], [LEFT_LOS_COL, RIGHT_LOS_COL]] = np.nan
    values = truth[list(VALUE_COLS)].fillna(10.0)
    values.loc[unscored, T07_POWER_COL] = 1e6

    power_only = values.copy()
    power_only[T07_POWER_COL] -= 2 * POWER_NORMALISER * truth[T03_POWER_COL]
    power_only.loc[unscored, T07_POWER_COL] = 1e6
    lr_only = values.copy()
    lr_only[LEFT_LOS_COL], lr_only[RIGHT_LOS_COL] = _l_r_values(20.0 + 2 * LR_NORMALISER_PCT)

    power_result = score_submission(validate_submission(_submission_csv_df(power_only)), truth)
    lr_result = score_submission(validate_submission(_submission_csv_df(lr_only)), truth)

    assert power_result["power_minutes"] == 1410
    assert power_result["lidar_minutes"] == 705
    assert power_result["score"] == pytest.approx(1.0)
    assert power_result["lr_rmse_pct_points"] == pytest.approx(0.0)
    assert lr_result["score"] == pytest.approx(1.0)
    assert lr_result["power_rmse"] == pytest.approx(0.0)


def test_power_error_is_relative_to_measured_t03_power() -> None:
    """A 100 kW error counts twice as much when T03 makes 500 kW as when it makes 1000 kW."""
    truth = _truth()
    truth.loc[truth.index[: len(truth) // 2], T03_POWER_COL] = 500.0
    values = truth[list(VALUE_COLS)].copy()
    values[T07_POWER_COL] += 100.0

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["power_rmse"] == pytest.approx(np.sqrt((0.2**2 + 0.1**2) / 2))


def test_score_needs_truth_for_every_submission_minute() -> None:
    """A truth file that doesn't cover the submission is an organiser error, not an unscored minute."""
    with pytest.raises(ValueError, match="truth is missing"):
        score_submission(validate_submission(_perfect_submission()), _truth().iloc[1:])
