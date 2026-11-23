"""Scoring for the WeDoWind wake steering challenge.

A submission is one CSV with a row for every minute of every scoring period (``submission_timestamps()``) and
columns ``SUBMISSION_COLS``: the minute start in UTC, T07 minute-mean power and the ZXTM 126 m range, 59 m
segment height Left and Right LOS speed minute means. ``scoring/submission_template.csv`` is a blank one.

The scorer picks the scored minutes from the truth file: T03 consensus wind direction and nacelle wind speed in
the scoring window, T03 and T07 operating normally all minute and T03 power above ``MIN_T03_POWER_KW``. The
LiDAR term also needs both measured LOS values. Per scored minute:

- power error = (predicted T07 - measured T07) / measured T03;
- L-R = 100 (|L| - |R|) / ((|L| + |R|) / 2), from predicted and from measured LOS; the error is their difference
  in percentage points.

``score = POWER_WEIGHT RMSE(power error) / POWER_NORMALISER + LR_WEIGHT RMSE(L-R error) / LR_NORMALISER_PCT``,
pooled over every period. Lower is better; about 1 is no better than a model that ignores steering.

Usage::

    python -m scripts.wedowind_wake_model_2026.scoring <submission.csv> <truth.parquet>

Regenerate the template with ``write_submission_template()``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hot_open.parquet_io import read_parquet

SCORING_PERIODS = tuple(
    (pd.Timestamp(start, tz="UTC"), pd.Timestamp(end_excl, tz="UTC"))
    for start, end_excl in (
        ("2026-09-01 02:00", "2026-09-01 11:00"),
        ("2026-09-04 00:00", "2026-09-04 08:00"),
        ("2026-09-07 07:00", "2026-09-07 23:00"),
        ("2026-10-04 00:00", "2026-10-04 05:00"),
        ("2026-11-14 13:00", "2026-11-14 22:00"),
    )
)
WD_RANGE_DEG = (206.0, 222.0)
WS_RANGE_MS = (6.0, 12.0)
MIN_T03_POWER_KW = 150.0
POWER_NORMALISER = 0.4
LR_NORMALISER_PCT = 30.0
POWER_WEIGHT = 0.5
LR_WEIGHT = 0.5

TIMESTAMP_COL = "timestamp_utc"
T07_POWER_COL = "T07_power_kW"
LEFT_LOS_COL = "zxtm_126m_left_los_ms"
RIGHT_LOS_COL = "zxtm_126m_right_los_ms"
VALUE_COLS = (T07_POWER_COL, LEFT_LOS_COL, RIGHT_LOS_COL)
SUBMISSION_COLS = (TIMESTAMP_COL, *VALUE_COLS)

T03_WD_COL = "T03_wind_direction_deg"
T03_WS_COL = "T03_wind_speed_ms"
T03_POWER_COL = "T03_power_kW"
T03_NORMAL_COL = "T03_normal"
T07_NORMAL_COL = "T07_normal"
TRUTH_COLS = (T03_WD_COL, T03_WS_COL, T03_POWER_COL, T03_NORMAL_COL, T07_NORMAL_COL, *VALUE_COLS)

TEMPLATE_PATH = Path(__file__).parent / "scoring" / "submission_template.csv"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def submission_timestamps() -> pd.DatetimeIndex:
    """Return the start of every minute of every scoring period, in UTC."""
    ranges = [pd.date_range(start, end_excl, freq="1min", inclusive="left") for start, end_excl in SCORING_PERIODS]
    return ranges[0].append(ranges[1:]).rename(TIMESTAMP_COL)


def submission_template() -> pd.DataFrame:
    """Return a blank submission: every timestamp, empty value columns."""
    return pd.DataFrame(index=submission_timestamps(), columns=list(VALUE_COLS), dtype=float)


def write_submission_template(path: Path = TEMPLATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = submission_template()
    template.index = template.index.strftime(TIMESTAMP_FORMAT).rename(TIMESTAMP_COL)
    template.to_csv(path, lineterminator="\n")


def validate_submission(submission: pd.DataFrame) -> pd.DataFrame:
    """Return the value columns of ``submission`` indexed by UTC minute in template order, or raise ``ValueError``.

    ``submission`` has the ``SUBMISSION_COLS`` as columns; extra columns are ignored. It is rejected if a column
    is missing, a timestamp is missing, duplicated or extra, any value is not finite, or an LOS speed is
    negative. The ZXTM reports LOS speed magnitudes; zero is allowed.
    """
    missing_cols = [col for col in SUBMISSION_COLS if col not in submission.columns]
    if missing_cols:
        msg = f"submission is missing columns {missing_cols}"
        raise ValueError(msg)
    timestamps = pd.DatetimeIndex(pd.to_datetime(submission[TIMESTAMP_COL], utc=True, format="ISO8601"))
    duplicated = timestamps[timestamps.duplicated()].unique()
    if len(duplicated):
        msg = f"submission has {len(duplicated)} duplicated timestamps, e.g. {duplicated[0]}"
        raise ValueError(msg)
    expected = submission_timestamps()
    missing = expected.difference(timestamps)
    extra = timestamps.difference(expected)
    if len(missing) or len(extra):
        msg = (
            f"submission has {len(missing)} missing timestamps {list(missing[:3])} "
            f"and {len(extra)} extra {list(extra[:3])}"
        )
        raise ValueError(msg)
    values = submission[list(VALUE_COLS)].apply(pd.to_numeric, errors="coerce").set_axis(timestamps)
    not_finite = ~np.isfinite(values.to_numpy(dtype=float))
    if not_finite.any():
        rows, cols = np.nonzero(not_finite)
        msg = (
            f"submission has {not_finite.sum()} missing or non-finite values, "
            f"e.g. {VALUE_COLS[cols[0]]} at {values.index[rows[0]]}"
        )
        raise ValueError(msg)
    negative = values[[LEFT_LOS_COL, RIGHT_LOS_COL]].lt(0).any(axis=1)
    if negative.any():
        msg = f"submission has {negative.sum()} minutes with a negative LOS speed, e.g. {negative.idxmax()}"
        raise ValueError(msg)
    return values.reindex(expected)


def read_submission(path: Path) -> pd.DataFrame:
    """Read and validate a submission CSV; see ``validate_submission``."""
    return validate_submission(pd.read_csv(path, dtype={TIMESTAMP_COL: str}))


def scored_minutes(truth: pd.DataFrame) -> pd.DataFrame:
    """Return boolean ``power`` and ``lidar`` columns marking the minutes of ``truth`` each score term uses.

    ``truth`` has the ``TRUTH_COLS``; missing values never count as in range or normal.
    """
    power = (
        truth[T03_WD_COL].ge(WD_RANGE_DEG[0])
        & truth[T03_WD_COL].lt(WD_RANGE_DEG[1])
        & truth[T03_WS_COL].ge(WS_RANGE_MS[0])
        & truth[T03_WS_COL].lt(WS_RANGE_MS[1])
        & truth[T03_NORMAL_COL].astype("boolean").fillna(value=False)
        & truth[T07_NORMAL_COL].astype("boolean").fillna(value=False)
        & truth[T03_POWER_COL].gt(MIN_T03_POWER_KW)
        & truth[T07_POWER_COL].notna()
    ).astype(bool)
    lidar = power & truth[LEFT_LOS_COL].notna() & truth[RIGHT_LOS_COL].notna()
    return pd.DataFrame({"power": power, "lidar": lidar}, index=truth.index)


def left_minus_right_pct(*, left: pd.Series, right: pd.Series) -> pd.Series:
    """Return 100 (|L| - |R|) / ((|L| + |R|) / 2), and 0 where both are zero."""
    mean = (left.abs() + right.abs()) / 2
    return (100 * (left.abs() - right.abs()) / mean.where(mean > 0)).fillna(0.0)


def _rmse(err: pd.Series) -> float:
    return float(np.sqrt((err**2).mean()))


def score_submission(submission: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int]:
    """Return the score, both RMSE terms and the scored minute counts.

    ``submission`` is as returned by ``validate_submission``; ``truth`` has the ``TRUTH_COLS``, indexed by UTC
    minute start, and must cover every submission timestamp.
    """
    truth = truth.copy()
    truth.index = pd.DatetimeIndex(truth.index).tz_convert("UTC")
    not_in_truth = submission.index.difference(truth.index)
    if len(not_in_truth):
        msg = f"truth is missing {len(not_in_truth)} submission timestamps, e.g. {not_in_truth[0]}"
        raise ValueError(msg)
    truth = truth.reindex(submission.index)
    mask = scored_minutes(truth)
    if not mask["power"].any() or not mask["lidar"].any():
        msg = f"no scored minutes: {mask.sum().to_dict()}"
        raise ValueError(msg)

    p, t = submission.loc[mask["power"]], truth.loc[mask["power"]]
    power_rmse = _rmse((p[T07_POWER_COL] - t[T07_POWER_COL]) / t[T03_POWER_COL])

    p, t = submission.loc[mask["lidar"]], truth.loc[mask["lidar"]]
    lr_rmse = _rmse(
        left_minus_right_pct(left=p[LEFT_LOS_COL], right=p[RIGHT_LOS_COL])
        - left_minus_right_pct(left=t[LEFT_LOS_COL], right=t[RIGHT_LOS_COL])
    )
    return {
        "score": POWER_WEIGHT * power_rmse / POWER_NORMALISER + LR_WEIGHT * lr_rmse / LR_NORMALISER_PCT,
        "power_rmse": power_rmse,
        "lr_rmse_pct_points": lr_rmse,
        "power_minutes": int(mask["power"].sum()),
        "lidar_minutes": int(mask["lidar"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a WeDoWind wake steering challenge submission.")
    parser.add_argument("submission", type=Path, help="submission CSV")
    parser.add_argument("truth", type=Path, help="truth parquet")
    args = parser.parse_args()
    result = score_submission(read_submission(args.submission), read_parquet(args.truth))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
