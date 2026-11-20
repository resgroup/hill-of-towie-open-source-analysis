"""Build the WeDoWind wake steering challenge truth file for the scoring periods.

The truth has one row per submission minute with the columns ``scoring.TRUTH_COLS``: T03 consensus wind
direction, nacelle wind speed and power, T03 and T07 normal-operation flags, T07 power and the ZXTM 126 m
hub-height Left and Right LOS minute means. Direction, speed, power and flags are as ``hot_floris.make_minute_df``
with T03 as the free stream; LOS minute means as ``zxtm_probes.zxtm_minute_means``.

It only runs where the scoring-day T03 and T07 fastlog and ZXTM files are available; they are not in the public
datapack, and the truth file is not published. The DY toggle-state tag is not needed. Fastlog tags are logged on
change, so each scoring day's fastlog needs the previous day too, to forward-fill across midnight, and the next
day's ``GenState_GenState`` and ``ActLimit_Power``, whose midnight values the loader reads to forward-fill them to
the end of the day; without them the normal-operation flags are False after the last change of either.

The fastlog cache is keyed without the filestore directory, so build the truth with a cache directory
(``HOT_OPEN_CACHE_DIR``) that has not cached these days from another filestore.

Usage::

    python -m scripts.wedowind_wake_model_2026.make_scoring_truth \
        --fastlog-dir <Filestore dir> [--fastlog-dir <another>] --lidar-dir <lidar_data dir> [--out <truth.parquet>]

Each ``--fastlog-dir`` holds ``FL/HOT/<serial>/<date>/``; a turbine is read from the first one that has it. The
defaults are the configured Filestore and ``<data dir>/lidar_data``.

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
    TI_WINDOW,
    WD_COL,
    YAW_POS_COL,
    check_data_present,
    make_minute_df,
)
from scripts.wedowind_wake_model_2026.scoring import (
    LEFT_LOS_COL,
    RIGHT_LOS_COL,
    SCORING_PERIODS,
    T03_NORMAL_COL,
    T03_POWER_COL,
    T03_WD_COL,
    T03_WS_COL,
    T07_NORMAL_COL,
    T07_POWER_COL,
    TRUTH_COLS,
    submission_timestamps,
)
from scripts.wedowind_wake_model_2026.zxtm_probes import HUB_HEIGHT_M, ZXTM_UNIT_ID, zxtm_minute_means

logger = logging.getLogger(__name__)

TRUTH_WTGS = ("T03", "T07")
LOS_RANGE_M = 126.0
SERIAL_OFFSET = 2304509


def fastlog_dir_for(wtg: str, *, fastlog_dirs: list[Path]) -> Path:
    serial = int(wtg[1:]) + SERIAL_OFFSET
    for fastlog_dir in fastlog_dirs:
        if (fastlog_dir / "FL" / "HOT" / str(serial)).is_dir():
            return fastlog_dir
    msg = f"no fastlog for {wtg} ({serial}) in {fastlog_dirs}"
    raise FileNotFoundError(msg)


def truth_for_period(
    *, start: pd.Timestamp, end_excl: pd.Timestamp, fastlog_dirs: list[Path], lidar_dir: Path
) -> pd.DataFrame:
    # padded so the direction smoothing and gap fill at the period edges see data either side
    load_start, load_end_excl = start - TI_WINDOW, end_excl + TI_WINDOW
    fl_df = pd.concat(
        [
            load_hot_fl_data(
                data_dir=fastlog_dir_for(wtg, fastlog_dirs=fastlog_dirs),
                wtg_numbers=[int(wtg[1:])],
                start_dt=load_start,
                end_dt_excl=load_end_excl,
                extra_tags=[WD_COL, YAW_POS_COL],
            )
            for wtg in TRUTH_WTGS
        ],
        axis=1,
    )
    check_data_present(fl_df, wtgs=list(TRUTH_WTGS), start_dt=start, end_dt_excl=end_excl)
    minute_df = make_minute_df(fl_df, free_stream_wtg="T03", wtgs=TRUTH_WTGS)

    zx_df = load_zx_lidar_fl_data(
        data_dir=lidar_dir, lidar_unit_id=ZXTM_UNIT_ID, start_dt=start, end_dt_excl=end_excl, remove_bad_values=True
    )
    if zx_df.empty:
        msg = f"no ZXTM {ZXTM_UNIT_ID} fastlog data from {start} to {end_excl}"
        raise ValueError(msg)
    los = zxtm_minute_means(zx_df)

    index = pd.date_range(start, end_excl, freq="1min", inclusive="left")
    return pd.DataFrame(
        {
            T03_WD_COL: minute_df["wd"],
            T03_WS_COL: minute_df["ws_raw"],
            T03_POWER_COL: minute_df["T03_power"],
            T03_NORMAL_COL: minute_df["T03_normal"],
            T07_NORMAL_COL: minute_df["T07_normal"],
            T07_POWER_COL: minute_df["T07_power"],
            LEFT_LOS_COL: los[(LOS_RANGE_M, HUB_HEIGHT_M, "Left")],
            RIGHT_LOS_COL: los[(LOS_RANGE_M, HUB_HEIGHT_M, "Right")],
        }
    ).reindex(index)


def make_scoring_truth(*, fastlog_dirs: list[Path], lidar_dir: Path) -> pd.DataFrame:
    """Return the truth for every scoring period, indexed like ``submission_timestamps()``."""
    truth = pd.concat(
        [
            truth_for_period(start=start, end_excl=end_excl, fastlog_dirs=fastlog_dirs, lidar_dir=lidar_dir)
            for start, end_excl in SCORING_PERIODS
        ]
    )
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
