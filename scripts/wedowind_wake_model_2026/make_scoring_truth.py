"""Build the WeDoWind wake steering challenge truth file for the scoring periods.

The truth has one row per submission minute with the columns ``scoring.TRUTH_COLS``: the pair, the upstream
turbine's consensus wind direction, nacelle wind speed and power, both turbines' normal-operation flags, the
downstream turbine's power and, for the LiDAR pair, the ZXTM 126 m hub-height Left and Right LOS minute means.
Direction, speed and power are as ``hot_floris.make_minute_df`` with the upstream turbine as the free stream; LOS
minute means as ``zxtm_probes.zxtm_minute_means``.

A turbine operates normally in a minute (``normal_minutes``) when it makes power every second and is never known to
be power limited: its power reduction flag ``PowerRed_PowerRed`` set, or ``ActLimit_Power`` below rated or, for
T11, whose ``ActLimit_Power`` follows its power in the scoring periods, less than ``MIN_ACT_LIMIT_HEADROOM_KW``
above its power. The loader stops forward-filling at any second without a power or wind speed sample, so on a quiet
turbine it drops ``ActLimit_Power`` and ``PowerRed_PowerRed``, which are logged on change, until their next change.
Each period is therefore loaded from midnight, where the loader has a value for both, and they are forward-filled
again here; a value still missing counts as not limited. Power is forward-filled across logging pauses up to
``POWER_FFILL_LIMIT_S``.

It only runs where the scoring-day fastlog and ZXTM files are available; they are not in the public datapack, and
the truth file is not published. The DY toggle-state tag is not needed. Fastlog tags are logged on change, so each
scoring day's fastlog needs the previous day too, to forward-fill across midnight, and the next day's
``ActLimit_Power`` and ``PowerRed_PowerRed``, whose midnight values the loader reads to forward-fill them to the end
of the day.

The fastlog cache is keyed without the filestore directory, so build the truth with a cache directory
(``HOT_OPEN_CACHE_DIR``) that has not cached these days from another filestore.

Usage::

    python -m scripts.wedowind_wake_model_2026.make_scoring_truth \
        --fastlog-dir <Filestore dir> [--fastlog-dir <another>] --lidar-dir <lidar_data dir> [--out <truth.parquet>]

Each ``--fastlog-dir`` holds ``FL/HOT/<serial>/<date>/``; a turbine's period is read from the first one that has
the turbine's folder for the period's start day. The defaults are the configured Filestore and
``<data dir>/lidar_data``.

Run with ``uv sync --group wedowind`` first.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_fl_data
from hot_open.settings import get_data_dir, get_filestore_dir, get_out_dir
from scripts.logger import setup_logger
from scripts.wedowind_wake_model_2026.hot_floris import (
    ACT_LIMIT_COL,
    POWER_COL,
    POWER_RED_COL,
    RATED_POWER_KW,
    SECONDS_PER_MINUTE,
    TI_WINDOW,
    WD_COL,
    YAW_POS_COL,
    check_data_present,
    make_minute_df,
)
from scripts.wedowind_wake_model_2026.scoring import (
    DOWNSTREAM_NORMAL_COL,
    DOWNSTREAM_POWER_COL,
    LEFT_LOS_COL,
    PAIR_COL,
    PAIRS,
    RIGHT_LOS_COL,
    TRUTH_COLS,
    UPSTREAM_NORMAL_COL,
    UPSTREAM_POWER_COL,
    UPSTREAM_WD_COL,
    UPSTREAM_WS_COL,
    TurbinePair,
    submission_timestamps,
)
from scripts.wedowind_wake_model_2026.zxtm_probes import HUB_HEIGHT_M, ZXTM_UNIT_ID, zxtm_minute_means

logger = logging.getLogger(__name__)

LOS_RANGE_M = 126.0
SERIAL_OFFSET = 2304509
HEADROOM_NORMAL_WTGS = frozenset({"T11"})
MIN_ACT_LIMIT_HEADROOM_KW = 50.0
POWER_FFILL_LIMIT_S = 5


def fastlog_dir_for(wtg: str, *, day: pd.Timestamp, fastlog_dirs: list[Path]) -> Path:
    serial = int(wtg[1:]) + SERIAL_OFFSET
    for fastlog_dir in fastlog_dirs:
        if (fastlog_dir / "FL" / "HOT" / str(serial) / f"{day:%Y-%m-%d}").is_dir():
            return fastlog_dir
    msg = f"no fastlog for {wtg} ({serial}) on {day:%Y-%m-%d} in {fastlog_dirs}"
    raise FileNotFoundError(msg)


def normal_minutes(wtg_fl_df: pd.DataFrame, *, wtg: str) -> pd.Series:
    """Return per minute whether the turbine made power every second and was never known to be power limited.

    ``wtg_fl_df`` is one turbine's 1 Hz fastlog with ``ActLimit_Power`` and ``PowerRed_PowerRed`` already
    forward-filled; a missing value of either counts as not limited.
    """
    act_limit, power = wtg_fl_df[ACT_LIMIT_COL], wtg_fl_df[POWER_COL]
    limited = wtg_fl_df[POWER_RED_COL] > 0
    if wtg in HEADROOM_NORMAL_WTGS:
        limited |= act_limit - power < MIN_ACT_LIMIT_HEADROOM_KW
    else:
        limited |= act_limit < RATED_POWER_KW
    normal = (power > 0) & ~limited
    return normal.resample(pd.Timedelta(minutes=1)).sum() == SECONDS_PER_MINUTE


def load_period_fl_df(
    wtgs: list[str], *, start: pd.Timestamp, end_excl: pd.Timestamp, fastlog_dirs: list[Path]
) -> pd.DataFrame:
    """Return the 1 Hz fastlog of ``wtgs`` for one scoring period, with the status tags and power forward-filled.

    It starts at the midnight before ``start`` and runs ``TI_WINDOW`` past ``end_excl``.
    """
    # padded so the direction smoothing and gap fill at the period edges see data either side, and from midnight
    # so the status tags have a value to forward-fill into the period
    load_start, load_end_excl = (start - TI_WINDOW).floor("D"), end_excl + TI_WINDOW
    fl_df = pd.concat(
        [
            load_hot_fl_data(
                data_dir=fastlog_dir_for(wtg, day=start, fastlog_dirs=fastlog_dirs),
                wtg_numbers=[int(wtg[1:])],
                start_dt=load_start,
                end_dt_excl=load_end_excl,
                extra_tags=[WD_COL, YAW_POS_COL],
            )
            for wtg in wtgs
        ],
        axis=1,
    )
    check_data_present(fl_df, wtgs=wtgs, start_dt=start, end_dt_excl=end_excl)
    for wtg in wtgs:
        for col in (ACT_LIMIT_COL, POWER_RED_COL):
            fl_df[(wtg, col)] = fl_df[(wtg, col)].ffill()
        fl_df[(wtg, POWER_COL)] = fl_df[(wtg, POWER_COL)].ffill(limit=POWER_FFILL_LIMIT_S)
    return fl_df


def truth_for_period(
    pair: TurbinePair,
    *,
    start: pd.Timestamp,
    end_excl: pd.Timestamp,
    fastlog_dirs: list[Path],
    lidar_dir: Path,
) -> pd.DataFrame:
    wtgs = [pair.upstream, pair.downstream]
    fl_df = load_period_fl_df(wtgs, start=start, end_excl=end_excl, fastlog_dirs=fastlog_dirs)
    minute_df = make_minute_df(fl_df, free_stream_wtg=pair.upstream, wtgs=wtgs)
    for wtg in wtgs:
        minute_df[f"{wtg}_normal"] = normal_minutes(fl_df[wtg], wtg=wtg)

    index = pd.date_range(start, end_excl, freq="1min", inclusive="left")
    truth = pd.DataFrame(
        {
            PAIR_COL: pair.name,
            UPSTREAM_WD_COL: minute_df["wd"],
            UPSTREAM_WS_COL: minute_df["ws_raw"],
            UPSTREAM_POWER_COL: minute_df[f"{pair.upstream}_power"],
            UPSTREAM_NORMAL_COL: minute_df[f"{pair.upstream}_normal"],
            DOWNSTREAM_NORMAL_COL: minute_df[f"{pair.downstream}_normal"],
            DOWNSTREAM_POWER_COL: minute_df[f"{pair.downstream}_power"],
            LEFT_LOS_COL: float("nan"),
            RIGHT_LOS_COL: float("nan"),
        }
    ).reindex(index)
    truth[PAIR_COL] = pair.name
    if pair.lidar:
        zx_df = load_zx_lidar_fl_data(
            data_dir=lidar_dir, lidar_unit_id=ZXTM_UNIT_ID, start_dt=start, end_dt_excl=end_excl, remove_bad_values=True
        )
        if zx_df.empty:
            msg = f"no ZXTM {ZXTM_UNIT_ID} fastlog data from {start} to {end_excl}"
            raise ValueError(msg)
        los = zxtm_minute_means(zx_df)
        truth[LEFT_LOS_COL] = los[(LOS_RANGE_M, HUB_HEIGHT_M, "Left")].reindex(index)
        truth[RIGHT_LOS_COL] = los[(LOS_RANGE_M, HUB_HEIGHT_M, "Right")].reindex(index)
    return truth


def make_scoring_truth(*, fastlog_dirs: list[Path], lidar_dir: Path) -> pd.DataFrame:
    """Return the truth for every scoring period, indexed like ``submission_timestamps()``."""
    truth = pd.concat(
        [
            truth_for_period(pair, start=start, end_excl=end_excl, fastlog_dirs=fastlog_dirs, lidar_dir=lidar_dir)
            for pair in PAIRS
            for start, end_excl in pair.periods
        ]
    ).sort_index()
    truth.index = truth.index.rename(submission_timestamps().name)
    return truth[list(TRUTH_COLS)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WeDoWind wake steering challenge truth file.")
    parser.add_argument("--fastlog-dir", type=Path, action="append", dest="fastlog_dirs")
    parser.add_argument("--lidar-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    setup_logger(out_dir / f"{Path(__file__).stem}.log")
    truth = make_scoring_truth(
        fastlog_dirs=args.fastlog_dirs or [get_filestore_dir()],
        lidar_dir=args.lidar_dir or get_data_dir() / "lidar_data",
    )
    out_path = args.out or out_dir / "scoring_truth.parquet"
    logger.info("Writing: %s", out_path)
    truth.to_parquet(out_path)


if __name__ == "__main__":
    main()
