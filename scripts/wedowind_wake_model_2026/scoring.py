"""Scoring for the WeDoWind wake steering challenge.

The challenge has four turbine pairs (``PAIRS``): an upstream turbine and a downstream turbine in its wake.
T03 is upstream of T07, T11 of T14 and T07 of T08 and T04. Each scoring period belongs to one pair.

A submission is one CSV with a row for every minute of every scoring period (``submission_timestamps()``) and
columns ``SUBMISSION_COLS``: the minute start in UTC, the pair, the downstream turbine's minute-mean power and,
for T03 -> T07 minutes only, the ZXTM 126 m range, 59 m segment height Left and Right LOS speed minute means. The
LOS columns are ignored, and may be blank, in the other pairs' minutes. ``scoring/submission_template.csv`` is a
blank one with the pair filled in.

The scorer picks the scored minutes from the truth file: upstream consensus wind direction in the pair's window,
upstream nacelle wind speed in ``WS_RANGE_MS``, both turbines operating normally all minute and upstream power
above ``MIN_UPSTREAM_POWER_KW``. The LiDAR term also needs both measured LOS values. Per scored minute:

- power error = (predicted downstream - measured downstream) / measured upstream;
- L-R = 100 (|L| - |R|) / ((|L| + |R|) / 2), from predicted and from measured LOS; the error is their difference
  in percentage points.

There is one power term per upstream turbine (T03, T11, T07), each the RMSE of the power error pooled over that
turbine's periods, and one L-R term::

    score = POWER_WEIGHT sum(RMSE(power error) / POWER_NORMALISER) + LR_WEIGHT RMSE(L-R error) / LR_NORMALISER_PCT

Lower is better. The normalisers are about the per-minute errors of a model that ignores steering on T03 -> T07,
so about 1 is no better than ignoring steering there.

Usage::

    python -m scripts.wedowind_wake_model_2026.scoring <submission.csv> <truth.parquet>

Regenerate the template with ``write_submission_template()``.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hot_open.parquet_io import read_parquet


@dataclass(frozen=True)
class TurbinePair:
    name: str
    upstream: str
    downstream: str
    wd_range_deg: tuple[float, float]
    periods: tuple[tuple[pd.Timestamp, pd.Timestamp], ...]
    lidar: bool = False


def _periods(*start_end_excl: tuple[str, str]) -> tuple[tuple[pd.Timestamp, pd.Timestamp], ...]:
    return tuple((pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")) for start, end in start_end_excl)


PAIRS = (
    TurbinePair(
        name="T03->T07",
        upstream="T03",
        downstream="T07",
        wd_range_deg=(206.0, 222.0),
        periods=_periods(
            ("2026-09-01 02:00", "2026-09-01 04:30"),
            ("2026-09-04 03:00", "2026-09-04 07:00"),
            ("2026-10-04 00:00", "2026-10-04 04:00"),
            ("2026-11-14 13:00", "2026-11-14 19:30"),
        ),
        lidar=True,
    ),
    TurbinePair(
        name="T11->T14",
        upstream="T11",
        downstream="T14",
        wd_range_deg=(228.0, 256.0),
        periods=_periods(
            ("2026-07-30 23:30", "2026-07-31 07:00"),
            ("2026-07-31 22:00", "2026-08-01 10:00"),
            ("2026-10-15 04:00", "2026-10-15 09:30"),
        ),
    ),
    TurbinePair(
        name="T07->T08",
        upstream="T07",
        downstream="T08",
        wd_range_deg=(260.0, 295.0),
        periods=_periods(("2026-10-10 20:00", "2026-10-11 00:00")),
    ),
    TurbinePair(
        name="T07->T04",
        upstream="T07",
        downstream="T04",
        wd_range_deg=(312.0, 342.0),
        periods=_periods(("2026-07-18 21:00", "2026-07-19 00:00"), ("2026-08-20 15:30", "2026-08-21 00:00")),
    ),
)
PAIRS_BY_NAME = {pair.name: pair for pair in PAIRS}
UPSTREAM_WTGS = tuple(dict.fromkeys(pair.upstream for pair in PAIRS))
# (start, end exclusive, pair name), in time order
SCORING_PERIODS = tuple(sorted((start, end, pair.name) for pair in PAIRS for start, end in pair.periods))

WS_RANGE_MS = (6.0, 12.0)
MIN_UPSTREAM_POWER_KW = 150.0
POWER_NORMALISER = 0.4
LR_NORMALISER_PCT = 30.0
POWER_WEIGHT = 0.25
LR_WEIGHT = 0.25

TIMESTAMP_COL = "timestamp_utc"
PAIR_COL = "pair"
DOWNSTREAM_POWER_COL = "downstream_power_kW"
LEFT_LOS_COL = "zxtm_126m_left_los_ms"
RIGHT_LOS_COL = "zxtm_126m_right_los_ms"
LOS_COLS = (LEFT_LOS_COL, RIGHT_LOS_COL)
VALUE_COLS = (DOWNSTREAM_POWER_COL, *LOS_COLS)
SUBMISSION_COLS = (TIMESTAMP_COL, PAIR_COL, *VALUE_COLS)

UPSTREAM_WD_COL = "upstream_wind_direction_deg"
UPSTREAM_WS_COL = "upstream_wind_speed_ms"
UPSTREAM_POWER_COL = "upstream_power_kW"
UPSTREAM_NORMAL_COL = "upstream_normal"
DOWNSTREAM_NORMAL_COL = "downstream_normal"
TRUTH_COLS = (
    PAIR_COL,
    UPSTREAM_WD_COL,
    UPSTREAM_WS_COL,
    UPSTREAM_POWER_COL,
    UPSTREAM_NORMAL_COL,
    DOWNSTREAM_NORMAL_COL,
    *VALUE_COLS,
)

TEMPLATE_PATH = Path(__file__).parent / "scoring" / "submission_template.csv"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def submission_pairs() -> pd.Series:
    """Return the pair name of every minute start of every scoring period, indexed by the UTC minute, in time order."""
    parts = [
        pd.Series(name, index=pd.date_range(start, end_excl, freq="1min", inclusive="left"))
        for start, end_excl, name in SCORING_PERIODS
    ]
    pairs = pd.concat(parts)
    pairs.index = pairs.index.rename(TIMESTAMP_COL)
    return pairs.rename(PAIR_COL)


def submission_timestamps() -> pd.DatetimeIndex:
    """Return the start of every minute of every scoring period, in UTC and time order."""
    return pd.DatetimeIndex(submission_pairs().index)


def submission_template() -> pd.DataFrame:
    """Return a blank submission: every timestamp, its pair, empty value columns."""
    template = submission_pairs().to_frame()
    for col in VALUE_COLS:
        template[col] = np.nan
    return template


def write_submission_template(path: Path = TEMPLATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    template = submission_template()
    template.index = template.index.strftime(TIMESTAMP_FORMAT).rename(TIMESTAMP_COL)
    template.to_csv(path, lineterminator="\n")


def _lidar_minutes(pairs: pd.Series) -> pd.Series:
    return pairs.map({pair.name: pair.lidar for pair in PAIRS}).astype(bool)


def validate_submission(submission: pd.DataFrame) -> pd.DataFrame:
    """Return the pair and value columns of ``submission`` indexed by UTC minute in template order, or raise
    ``ValueError``.

    ``submission`` has the ``SUBMISSION_COLS`` as columns; extra columns are ignored. It is rejected if a column
    is missing, a timestamp is missing, duplicated or extra, a pair differs from the template's, a downstream power
    is not finite, or, in a LiDAR pair's minutes, an LOS speed is not finite or is negative. The ZXTM reports LOS
    speed magnitudes; zero is allowed. LOS values in other pairs' minutes are returned as NaN.
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
    expected_pairs = submission_pairs()
    expected = pd.DatetimeIndex(expected_pairs.index)
    missing = expected.difference(timestamps)
    extra = timestamps.difference(expected)
    if len(missing) or len(extra):
        msg = (
            f"submission has {len(missing)} missing timestamps {list(missing[:3])} "
            f"and {len(extra)} extra {list(extra[:3])}"
        )
        raise ValueError(msg)
    pairs = submission[PAIR_COL].astype(str).set_axis(timestamps).reindex(expected)
    wrong_pair = pairs.ne(expected_pairs.to_numpy())
    if wrong_pair.any():
        msg = (
            f"submission has {wrong_pair.sum()} minutes with the wrong pair, e.g. {pairs[wrong_pair].iloc[0]!r} "
            f"at {wrong_pair.idxmax()} where the template has {expected_pairs[wrong_pair].iloc[0]!r}"
        )
        raise ValueError(msg)
    values = submission[list(VALUE_COLS)].apply(pd.to_numeric, errors="coerce").set_axis(timestamps).reindex(expected)
    lidar = _lidar_minutes(expected_pairs).to_numpy()
    values.loc[~lidar, list(LOS_COLS)] = np.nan
    required = values.to_numpy(dtype=float)
    required_mask = np.column_stack([np.ones(len(values), dtype=bool), lidar, lidar])
    not_finite = required_mask & ~np.isfinite(required)
    if not_finite.any():
        rows, cols = np.nonzero(not_finite)
        msg = (
            f"submission has {not_finite.sum()} missing or non-finite values, "
            f"e.g. {VALUE_COLS[cols[0]]} at {values.index[rows[0]]}"
        )
        raise ValueError(msg)
    negative = values[list(LOS_COLS)].lt(0).any(axis=1)
    if negative.any():
        msg = f"submission has {negative.sum()} minutes with a negative LOS speed, e.g. {negative.idxmax()}"
        raise ValueError(msg)
    return pd.concat([pairs, values], axis=1)


def read_submission(path: Path) -> pd.DataFrame:
    """Read and validate a submission CSV; see ``validate_submission``."""
    return validate_submission(pd.read_csv(path, dtype={TIMESTAMP_COL: str, PAIR_COL: str}))


def scored_minutes(truth: pd.DataFrame) -> pd.DataFrame:
    """Return boolean ``power`` and ``lidar`` columns marking the minutes of ``truth`` each score term uses.

    ``truth`` has the ``TRUTH_COLS``; the direction window is the minute's pair's. Missing values never count as
    in range or normal, and only LiDAR pairs' minutes count for ``lidar``.
    """
    lo = truth[PAIR_COL].map({pair.name: pair.wd_range_deg[0] for pair in PAIRS})
    hi = truth[PAIR_COL].map({pair.name: pair.wd_range_deg[1] for pair in PAIRS})
    power = (
        truth[UPSTREAM_WD_COL].ge(lo)
        & truth[UPSTREAM_WD_COL].lt(hi)
        & truth[UPSTREAM_WS_COL].ge(WS_RANGE_MS[0])
        & truth[UPSTREAM_WS_COL].lt(WS_RANGE_MS[1])
        & truth[UPSTREAM_NORMAL_COL].astype("boolean").fillna(value=False)
        & truth[DOWNSTREAM_NORMAL_COL].astype("boolean").fillna(value=False)
        & truth[UPSTREAM_POWER_COL].gt(MIN_UPSTREAM_POWER_KW)
        & truth[DOWNSTREAM_POWER_COL].notna()
    ).astype(bool)
    lidar = power & _lidar_minutes(truth[PAIR_COL]) & truth[LEFT_LOS_COL].notna() & truth[RIGHT_LOS_COL].notna()
    return pd.DataFrame({"power": power, "lidar": lidar}, index=truth.index)


def left_minus_right_pct(*, left: pd.Series, right: pd.Series) -> pd.Series:
    """Return 100 (|L| - |R|) / ((|L| + |R|) / 2), and 0 where both are zero."""
    mean = (left.abs() + right.abs()) / 2
    return (100 * (left.abs() - right.abs()) / mean.where(mean > 0)).fillna(0.0)


def _rmse(err: pd.Series) -> float:
    return float(np.sqrt((err**2).mean()))


def score_submission(submission: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float | int]:
    """Return the score, every RMSE term and its scored minute count.

    ``submission`` is as returned by ``validate_submission``; ``truth`` has the ``TRUTH_COLS``, indexed by UTC
    minute start, and must cover every submission timestamp with the same pair. The power terms are keyed by
    upstream turbine, e.g. ``power_rmse_T03`` and ``power_minutes_T03``.
    """
    truth = truth.copy()
    truth.index = pd.DatetimeIndex(truth.index).tz_convert("UTC")
    not_in_truth = submission.index.difference(truth.index)
    if len(not_in_truth):
        msg = f"truth is missing {len(not_in_truth)} submission timestamps, e.g. {not_in_truth[0]}"
        raise ValueError(msg)
    truth = truth.reindex(submission.index)
    if not truth[PAIR_COL].eq(submission[PAIR_COL]).all():
        msg = "truth and submission pairs differ"
        raise ValueError(msg)
    mask = scored_minutes(truth)
    upstream = truth[PAIR_COL].map({pair.name: pair.upstream for pair in PAIRS})

    result: dict[str, float | int] = {}
    score = 0.0
    for wtg in UPSTREAM_WTGS:
        scored = mask["power"] & upstream.eq(wtg)
        if not scored.any():
            msg = f"no scored power minutes for {wtg}"
            raise ValueError(msg)
        p, t = submission.loc[scored], truth.loc[scored]
        rmse = _rmse((p[DOWNSTREAM_POWER_COL] - t[DOWNSTREAM_POWER_COL]) / t[UPSTREAM_POWER_COL])
        score += POWER_WEIGHT * rmse / POWER_NORMALISER
        result[f"power_rmse_{wtg}"] = rmse
        result[f"power_minutes_{wtg}"] = int(scored.sum())

    if not mask["lidar"].any():
        msg = "no scored LiDAR minutes"
        raise ValueError(msg)
    p, t = submission.loc[mask["lidar"]], truth.loc[mask["lidar"]]
    lr_rmse = _rmse(
        left_minus_right_pct(left=p[LEFT_LOS_COL], right=p[RIGHT_LOS_COL])
        - left_minus_right_pct(left=t[LEFT_LOS_COL], right=t[RIGHT_LOS_COL])
    )
    score += LR_WEIGHT * lr_rmse / LR_NORMALISER_PCT
    result["lr_rmse_pct_points"] = lr_rmse
    result["lidar_minutes"] = int(mask["lidar"].sum())
    return {"score": score, **result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a WeDoWind wake steering challenge submission.")
    parser.add_argument("submission", type=Path, help="submission CSV")
    parser.add_argument("truth", type=Path, help="truth parquet")
    args = parser.parse_args()
    result = score_submission(read_submission(args.submission), read_parquet(args.truth))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
