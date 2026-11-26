from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.wedowind_wake_model_2026.scoring import (
    DOWNSTREAM_NORMAL_COL,
    DOWNSTREAM_POWER_COL,
    LEFT_LOS_COL,
    LOS_COLS,
    LR_NORMALISER_PCT,
    MIN_UPSTREAM_POWER_KW,
    NO_STEERING_POWER_COL,
    PAIR_COL,
    PAIRS,
    PAIRS_BY_NAME,
    POWER_NORMALISER,
    RIGHT_LOS_COL,
    SCORING_PERIODS,
    STEERING_WTGS,
    SUBMISSION_COLS,
    TEMPLATE_PATH,
    TIMESTAMP_COL,
    TRUTH_COLS,
    UPSTREAM_NORMAL_COL,
    UPSTREAM_POWER_COL,
    UPSTREAM_WD_COL,
    UPSTREAM_WS_COL,
    VALUE_COLS,
    WS_RANGE_MS,
    read_submission,
    score_submission,
    scored_minutes,
    submission_pairs,
    submission_template,
    submission_timestamps,
    validate_submission,
    write_submission_template,
)

N_MINUTES = 3450
LIDAR_PAIR = "T03->T07"


def _truth() -> pd.DataFrame:
    """Every minute scored: mid-window direction, both turbines normal, T03 -> T07 L-R of +20 % (L 11, R 9)."""
    pairs = submission_pairs()
    mid_wd = pairs.map({pair.name: sum(pair.wd_range_deg) / 2 for pair in PAIRS})
    lidar = pairs.eq(LIDAR_PAIR)
    return pd.DataFrame(
        {
            PAIR_COL: pairs,
            UPSTREAM_WD_COL: mid_wd,
            UPSTREAM_WS_COL: 9.0,
            UPSTREAM_POWER_COL: 1000.0,
            UPSTREAM_NORMAL_COL: True,
            DOWNSTREAM_NORMAL_COL: True,
            DOWNSTREAM_POWER_COL: 700.0,
            LEFT_LOS_COL: np.where(lidar, 11.0, np.nan),
            RIGHT_LOS_COL: np.where(lidar, 9.0, np.nan),
        },
        index=pairs.index,
    )[list(TRUTH_COLS)]


def _submission_csv_df(values: pd.DataFrame) -> pd.DataFrame:
    return values.reset_index(names=TIMESTAMP_COL).assign(
        **{TIMESTAMP_COL: lambda df: df[TIMESTAMP_COL].dt.strftime("%Y-%m-%dT%H:%M:%SZ")}
    )


def _values(truth: pd.DataFrame) -> pd.DataFrame:
    return truth[[PAIR_COL, *VALUE_COLS]].copy()


def _perfect_submission() -> pd.DataFrame:
    return _submission_csv_df(_values(_truth()))


def _row(sub: pd.DataFrame, pair: str, n: int = 0) -> int:
    return int(np.flatnonzero(sub[PAIR_COL].eq(pair))[n])


def test_pairs_cover_three_steering_turbines_and_four_downstream_turbines() -> None:
    """T03 -> T07 carries the LiDAR term; T07 steers for both T08 and T04."""
    assert STEERING_WTGS == ("T03", "T11", "T07")
    assert {pair.downstream for pair in PAIRS} == {"T07", "T14", "T08", "T04"}
    assert [pair.name for pair in PAIRS if pair.lidar] == [LIDAR_PAIR]
    assert {name: len(PAIRS_BY_NAME[name].periods) for name in PAIRS_BY_NAME} == {
        "T03->T07": 4,
        "T11->T14": 3,
        "T07->T08": 1,
        "T07->T04": 2,
    }


def test_template_timestamps_are_every_minute_start_of_every_period_end_exclusive() -> None:
    """3,450 minute starts in time order; each period starts on its start and stops a minute before its end."""
    timestamps = submission_timestamps()
    pairs = submission_pairs()

    assert len(timestamps) == N_MINUTES
    assert timestamps.is_unique
    assert timestamps.is_monotonic_increasing
    assert str(timestamps.tz) == "UTC"
    assert (timestamps.second == 0).all()
    for start, end_excl, name in SCORING_PERIODS:
        assert start in timestamps
        assert end_excl not in timestamps
        assert end_excl - pd.Timedelta(minutes=1) in timestamps
        assert (pairs[(pairs.index >= start) & (pairs.index < end_excl)] == name).all()
    assert pairs.value_counts().to_dict() == {"T11->T14": 1500, "T03->T07": 1020, "T07->T04": 690, "T07->T08": 240}


def test_committed_template_matches_submission_timestamps(tmp_path: Path) -> None:
    """The committed blank template is what ``write_submission_template`` writes, so it can't drift."""
    regenerated = tmp_path / "template.csv"
    write_submission_template(regenerated)

    assert TEMPLATE_PATH.read_bytes() == regenerated.read_bytes()
    template = pd.read_csv(TEMPLATE_PATH)
    assert list(template.columns) == list(SUBMISSION_COLS)
    assert template.loc[0, TIMESTAMP_COL] == "2026-07-18T21:00:00Z"
    assert template.loc[0, PAIR_COL] == "T07->T04"
    assert template[[*VALUE_COLS, NO_STEERING_POWER_COL]].isna().all().all()
    assert submission_template().shape == (N_MINUTES, 5)


def test_read_submission_round_trips_a_csv(tmp_path: Path) -> None:
    """Rows may come in any order; the values come back in template order."""
    path = tmp_path / "submission.csv"
    _perfect_submission().sample(frac=1, random_state=0).to_csv(path, index=False)

    values = read_submission(path)

    assert values.index.equals(submission_timestamps())
    pd.testing.assert_frame_equal(values[[PAIR_COL, *VALUE_COLS]], _values(_truth()), check_names=False)
    assert values[NO_STEERING_POWER_COL].isna().all()


def test_rejects_missing_duplicated_or_extra_timestamps() -> None:
    """The submission must have exactly the template timestamps, once each."""
    sub = _perfect_submission()
    with pytest.raises(ValueError, match="missing timestamps"):
        validate_submission(sub.drop(index=5))
    with pytest.raises(ValueError, match="duplicated"):
        validate_submission(pd.concat([sub, sub.iloc[[5]]]))
    extra = sub.iloc[[0]].assign(**{TIMESTAMP_COL: "2026-09-01T04:30:00Z"})
    with pytest.raises(ValueError, match="1 extra"):
        validate_submission(pd.concat([sub, extra]))


@pytest.mark.parametrize("col", [TIMESTAMP_COL, PAIR_COL, *VALUE_COLS])
def test_rejects_a_missing_column(col: str) -> None:
    """Each scored column, the timestamp and the pair are required."""
    with pytest.raises(ValueError, match="missing columns"):
        validate_submission(_perfect_submission().drop(columns=col))


def test_rejects_a_wrong_pair() -> None:
    """The pair column must match the template, so a shifted or mislabelled row is caught."""
    sub = _perfect_submission()
    sub.loc[_row(sub, "T11->T14"), PAIR_COL] = "T07->T08"
    with pytest.raises(ValueError, match="wrong pair"):
        validate_submission(sub)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, "fast"])
@pytest.mark.parametrize("pair", [pair.name for pair in PAIRS])
def test_rejects_non_finite_downstream_power(pair: str, bad: float | str) -> None:
    """Every row's downstream power must be predicted, even minutes that won't be scored."""
    sub = _perfect_submission().astype({DOWNSTREAM_POWER_COL: object})
    sub.loc[_row(sub, pair, 7), DOWNSTREAM_POWER_COL] = bad
    with pytest.raises(ValueError, match="non-finite"):
        validate_submission(sub)


@pytest.mark.parametrize("bad", [np.nan, np.inf, "fast"])
@pytest.mark.parametrize("col", LOS_COLS)
def test_rejects_non_finite_los_in_the_lidar_pair(col: str, bad: float | str) -> None:
    """Every T03 -> T07 row must have both LOS speeds."""
    sub = _perfect_submission().astype({col: object})
    sub.loc[_row(sub, LIDAR_PAIR, 7), col] = bad
    with pytest.raises(ValueError, match="non-finite"):
        validate_submission(sub)


def test_los_is_ignored_outside_the_lidar_pair() -> None:
    """Other pairs' LOS columns may be blank, or anything, and come back as NaN."""
    sub = _perfect_submission().astype({LEFT_LOS_COL: object, RIGHT_LOS_COL: object})
    other = sub[PAIR_COL].ne(LIDAR_PAIR)
    sub.loc[other, LEFT_LOS_COL] = "n/a"
    sub.loc[other, RIGHT_LOS_COL] = -5.0

    values = validate_submission(sub)

    assert values.loc[values[PAIR_COL].ne(LIDAR_PAIR), list(LOS_COLS)].isna().all().all()
    assert values.loc[values[PAIR_COL].eq(LIDAR_PAIR), list(LOS_COLS)].notna().all().all()


@pytest.mark.parametrize("col", LOS_COLS)
def test_rejects_a_negative_los_speed(col: str) -> None:
    """The ZXTM reports LOS speed magnitudes, so a negative one is not a valid prediction."""
    sub = _perfect_submission()
    sub.loc[_row(sub, LIDAR_PAIR, 3), col] = -0.1
    with pytest.raises(ValueError, match="negative LOS speed"):
        validate_submission(sub)


def test_zero_los_speeds_are_accepted_and_scored_without_dividing_by_zero() -> None:
    """Calm is physically possible: both LOS zero gives L-R 0, scored against the measured +20 %."""
    sub = _perfect_submission()
    sub.loc[sub[PAIR_COL].eq(LIDAR_PAIR), list(LOS_COLS)] = 0.0

    result = score_submission(validate_submission(sub), _truth())

    assert result["lr_rmse_pct_points"] == pytest.approx(20.0)
    assert np.isfinite(result["score"])


def test_measured_zero_los_speeds_do_not_divide_by_zero() -> None:
    """The truth gets the same protection: both measured LOS zero gives a measured L-R of 0."""
    truth = _truth()
    truth.loc[truth[PAIR_COL].eq(LIDAR_PAIR), list(LOS_COLS)] = 0.0

    result = score_submission(validate_submission(_perfect_submission()), truth)

    assert result["lr_rmse_pct_points"] == pytest.approx(20.0)


def _one_minute_truth(pair: str) -> pd.DataFrame:
    truth = _truth()
    return truth.loc[truth[PAIR_COL].eq(pair)].iloc[[0]].copy()


def _edge_cases() -> list[tuple[str, str, object, bool]]:
    cases = []
    for pair in PAIRS:
        lo, hi = pair.wd_range_deg
        cases += [
            (pair.name, UPSTREAM_WD_COL, lo, True),
            (pair.name, UPSTREAM_WD_COL, lo - 1e-6, False),
            (pair.name, UPSTREAM_WD_COL, hi - 1e-6, True),
            (pair.name, UPSTREAM_WD_COL, hi, False),
        ]
    return [
        *cases,
        (LIDAR_PAIR, UPSTREAM_WD_COL, np.nan, False),
        (LIDAR_PAIR, UPSTREAM_WS_COL, WS_RANGE_MS[0], True),
        (LIDAR_PAIR, UPSTREAM_WS_COL, WS_RANGE_MS[0] - 1e-6, False),
        (LIDAR_PAIR, UPSTREAM_WS_COL, WS_RANGE_MS[1] - 1e-6, True),
        (LIDAR_PAIR, UPSTREAM_WS_COL, WS_RANGE_MS[1], False),
        (LIDAR_PAIR, UPSTREAM_WS_COL, np.nan, False),
        (LIDAR_PAIR, UPSTREAM_POWER_COL, MIN_UPSTREAM_POWER_KW, False),
        (LIDAR_PAIR, UPSTREAM_POWER_COL, MIN_UPSTREAM_POWER_KW + 1e-6, True),
        (LIDAR_PAIR, UPSTREAM_NORMAL_COL, False, False),
        (LIDAR_PAIR, UPSTREAM_NORMAL_COL, pd.NA, False),
        (LIDAR_PAIR, DOWNSTREAM_NORMAL_COL, False, False),
        (LIDAR_PAIR, DOWNSTREAM_NORMAL_COL, pd.NA, False),
        (LIDAR_PAIR, DOWNSTREAM_POWER_COL, np.nan, False),
    ]


@pytest.mark.parametrize(("pair", "col", "value", "scored"), _edge_cases())
def test_scored_minutes_window_edges(pair: str, col: str, value: object, *, scored: bool) -> None:
    """Direction (per pair) and speed windows are [lo, hi); upstream power must exceed its minimum.

    Missing values never count.
    """
    truth = _one_minute_truth(pair)
    truth[col] = pd.Series([value], index=truth.index, dtype="boolean" if "normal" in col else float)

    mask = scored_minutes(truth)

    assert mask["power"].iloc[0] == scored
    assert mask["lidar"].iloc[0] == (scored and pair == LIDAR_PAIR)


def test_direction_windows_are_per_pair() -> None:
    """A direction inside one pair's window is outside the others'."""
    truth = _truth()
    truth[UPSTREAM_WD_COL] = 214.0

    mask = scored_minutes(truth)

    assert mask["power"].groupby(truth[PAIR_COL]).all().to_dict() == {
        "T03->T07": True,
        "T07->T04": False,
        "T07->T08": False,
        "T11->T14": False,
    }


@pytest.mark.parametrize("col", LOS_COLS)
def test_lidar_term_needs_both_measured_los(col: str) -> None:
    """A minute missing either measured LOS still counts for power but not for L-R."""
    truth = _one_minute_truth(LIDAR_PAIR)
    truth[col] = np.nan

    mask = scored_minutes(truth)

    assert mask["power"].iloc[0]
    assert not mask["lidar"].iloc[0]


def test_perfect_submission_scores_zero() -> None:
    """Submitting the truth scores 0 with every minute scored."""
    result = score_submission(validate_submission(_perfect_submission()), _truth())

    assert result == {
        "score": 0.0,
        "power_rmse_T03": 0.0,
        "power_minutes_T03": 1020,
        "power_rmse_T11": 0.0,
        "power_minutes_T11": 1500,
        "power_rmse_T07": 0.0,
        "power_minutes_T07": 930,
        "lr_rmse_pct_points": 0.0,
        "lidar_minutes": 1020,
    }


def _l_r_values(lr_pct: float, *, mean: float = 10.0) -> tuple[float, float]:
    """Return (L, R) with mean ``mean`` and 100 (L - R) / mean = ``lr_pct``."""
    half_diff = lr_pct * mean / 200
    return mean + half_diff, mean - half_diff


def test_no_steering_sized_errors_score_one() -> None:
    """A power error of the power normaliser and an L-R error of the L-R normaliser every minute score 1."""
    truth = _truth()
    values = _values(truth)
    values[DOWNSTREAM_POWER_COL] += POWER_NORMALISER * truth[UPSTREAM_POWER_COL]
    lidar = values[PAIR_COL].eq(LIDAR_PAIR)
    left, right = _l_r_values(20.0 - LR_NORMALISER_PCT)
    values.loc[lidar, LEFT_LOS_COL], values.loc[lidar, RIGHT_LOS_COL] = left, right

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    for wtg in STEERING_WTGS:
        assert result[f"power_rmse_{wtg}"] == pytest.approx(POWER_NORMALISER)
    assert result["lr_rmse_pct_points"] == pytest.approx(LR_NORMALISER_PCT)
    assert result["score"] == pytest.approx(1.0)


@pytest.mark.parametrize("wtg", ["T03", "T11", "T07"])
def test_each_steering_turbine_power_term_weighs_a_quarter(wtg: str) -> None:
    """An error of the normaliser in one steering turbine's minutes only scores 0.25."""
    truth = _truth()
    values = _values(truth)
    upstream = truth[PAIR_COL].map({pair.name: pair.upstream for pair in PAIRS})
    values.loc[upstream.eq(wtg), DOWNSTREAM_POWER_COL] += POWER_NORMALISER * 1000.0

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["score"] == pytest.approx(0.25)
    assert result[f"power_rmse_{wtg}"] == pytest.approx(POWER_NORMALISER)


def test_lr_term_weighs_a_quarter() -> None:
    """An L-R error of the normaliser in every T03 -> T07 minute only scores 0.25."""
    truth = _truth()
    values = _values(truth)
    lidar = values[PAIR_COL].eq(LIDAR_PAIR)
    left, right = _l_r_values(20.0 + LR_NORMALISER_PCT)
    values.loc[lidar, LEFT_LOS_COL], values.loc[lidar, RIGHT_LOS_COL] = left, right

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["score"] == pytest.approx(0.25)


def test_t07_term_pools_both_of_its_downstream_turbines() -> None:
    """T07 -> T08 and T07 -> T04 minutes share one RMSE, so each minute weighs the same whichever turbine it is."""
    truth = _truth()
    values = _values(truth)
    values.loc[values[PAIR_COL].eq("T07->T08"), DOWNSTREAM_POWER_COL] += 100.0

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["power_minutes_T07"] == 930
    assert result["power_rmse_T07"] == pytest.approx(np.sqrt(240 / 930) * 0.1)


def test_unscored_minutes_do_not_count() -> None:
    """Minutes outside the speed window, or missing LOS, don't count however wrong the prediction."""
    truth = _truth()
    unscored = truth.index[::2]
    truth.loc[unscored, UPSTREAM_WS_COL] = 3.0
    lidar_rows = truth.index[truth[PAIR_COL].eq(LIDAR_PAIR)]
    truth.loc[lidar_rows[1::4], list(LOS_COLS)] = np.nan
    values = _values(truth)
    values.loc[unscored, DOWNSTREAM_POWER_COL] = 1e6
    values[list(LOS_COLS)] = values[list(LOS_COLS)].fillna(10.0)
    values.loc[values[PAIR_COL].ne(LIDAR_PAIR), list(LOS_COLS)] = np.nan

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["score"] == pytest.approx(0.0)
    assert result["power_minutes_T03"] + result["power_minutes_T11"] + result["power_minutes_T07"] == N_MINUTES // 2
    assert result["lidar_minutes"] < result["power_minutes_T03"]


def test_power_error_is_relative_to_measured_upstream_power() -> None:
    """A 100 kW error counts twice as much when the upstream turbine makes 500 kW as when it makes 1000 kW."""
    truth = _truth()
    t11 = truth.index[truth[PAIR_COL].eq("T11->T14")]
    truth.loc[t11[: len(t11) // 2], UPSTREAM_POWER_COL] = 500.0
    values = _values(truth)
    values[DOWNSTREAM_POWER_COL] += 100.0

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["power_rmse_T11"] == pytest.approx(np.sqrt((0.2**2 + 0.1**2) / 2))
    assert result["power_rmse_T03"] == pytest.approx(0.1)


def test_score_needs_truth_for_every_submission_minute() -> None:
    """A truth file that doesn't cover the submission is an organiser error, not an unscored minute."""
    with pytest.raises(ValueError, match="truth is missing"):
        score_submission(validate_submission(_perfect_submission()), _truth().iloc[1:])


def test_score_needs_truth_and_submission_pairs_to_agree() -> None:
    """A truth built for different periods is an organiser error."""
    truth = _truth()
    truth.loc[truth.index[0], PAIR_COL] = "T11->T14"
    with pytest.raises(ValueError, match="pairs differ"):
        score_submission(validate_submission(_perfect_submission()), truth)


@pytest.mark.parametrize("blank", [np.nan, "", " "])
def test_no_steering_power_may_be_left_blank(blank: float | str) -> None:
    """The template's blank no-steering column must validate and score as if the column were not there."""
    sub = _perfect_submission().assign(**{NO_STEERING_POWER_COL: blank})

    values = validate_submission(sub)

    assert values[NO_STEERING_POWER_COL].isna().all()
    assert not any(key.startswith("steering_gain") for key in score_submission(values, _truth()))


@pytest.mark.parametrize("bad", [np.nan, np.inf, "n/a"])
def test_rejects_a_partly_filled_no_steering_power(bad: float | str) -> None:
    """A gain from some minutes only would be misleading, so the column is all or nothing."""
    sub = _perfect_submission().assign(**{NO_STEERING_POWER_COL: 650.0}).astype({NO_STEERING_POWER_COL: object})
    sub.loc[_row(sub, "T11->T14"), NO_STEERING_POWER_COL] = bad

    with pytest.raises(ValueError, match=NO_STEERING_POWER_COL):
        validate_submission(sub)


def test_steering_gain_is_reported_per_steering_turbine_and_does_not_change_the_score() -> None:
    """The gain is the summed power over the summed no-steering power on scored minutes, outside the score."""
    truth = _truth()
    values = _values(truth)
    upstream = truth[PAIR_COL].map({pair.name: pair.upstream for pair in PAIRS})
    values[NO_STEERING_POWER_COL] = upstream.map({"T03": 700.0 / 1.05, "T11": 700.0, "T07": 700.0 / 0.98})
    unscored = values.index[_row(values, "T03->T07")]
    truth.loc[unscored, UPSTREAM_NORMAL_COL] = False
    values.loc[unscored, NO_STEERING_POWER_COL] = 1.0

    result = score_submission(validate_submission(_submission_csv_df(values)), truth)

    assert result["score"] == 0.0
    assert result["steering_gain_pct_T03"] == pytest.approx(5.0)
    assert result["steering_gain_pct_T11"] == pytest.approx(0.0)
    assert result["steering_gain_pct_T07"] == pytest.approx(-2.0)
