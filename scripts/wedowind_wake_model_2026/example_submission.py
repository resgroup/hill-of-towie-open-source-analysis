"""Worked example: a WeDoWind wake steering challenge submission from the public FLORIS model.

For every scoring period it uses only the upstream turbine's fastlog, as a participant has:

- free stream per minute: ``k`` times the upstream turbine's nacelle wind speed, its DY consensus wind direction
  and a centred 10-minute TI from its 1 Hz wind speed (``hot_floris.make_minute_df``);
- the upstream turbine gets its measured yaw misalignment and every other turbine zero yaw;
- ``k`` is fitted per upstream turbine for zero mean bias on its own power, over the minutes of its periods that
  pass the published scoring filter on the upstream turbine;
- the downstream power is FLORIS's power for the pair's downstream turbine;
- for T03 -> T07, the ZXTM LOS speeds are FLORIS sampled at the 126 m range, 59 m segment height Left and Right
  probes (``zxtm_probes.py``). T07's yaw position is not known on scoring days, so the ZXTM is assumed to point
  into T03's wind direction.

Minutes with a missing input are filled from their neighbours (zero for a missing yaw misalignment), because a
submission needs a value for every minute.

Usage::

    python -m scripts.wedowind_wake_model_2026.example_submission --fastlog-dir <Filestore dir> [--out <csv>]

``--fastlog-dir`` holds the scoring-day inputs as ``FL/HOT/<serial>/<date>/``; the default is the configured
Filestore.

Run with ``uv sync --group wedowind`` first.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from floris import FlorisModel

from hot_open.settings import get_filestore_dir, get_out_dir
from hot_open.unpack import unpack_local_meta_data
from scripts.logger import setup_logger
from scripts.wedowind_wake_model_2026.hot_floris import (
    FREE_STREAM_COLS,
    fit_ws_scale,
    load_hot_floris_model,
    make_minute_df,
    run_floris_minutes,
    yaw_angles_for,
)
from scripts.wedowind_wake_model_2026.make_scoring_truth import LOS_RANGE_M, load_period_fl_df, normal_minutes
from scripts.wedowind_wake_model_2026.scoring import (
    DOWNSTREAM_POWER_COL,
    LEFT_LOS_COL,
    MIN_UPSTREAM_POWER_KW,
    PAIR_COL,
    PAIRS,
    RIGHT_LOS_COL,
    TIMESTAMP_COL,
    TIMESTAMP_FORMAT,
    UPSTREAM_WTGS,
    WS_RANGE_MS,
    TurbinePair,
    submission_template,
    validate_submission,
)
from scripts.wedowind_wake_model_2026.zxtm_probes import HUB_HEIGHT_M, modelled_los_minutes

logger = logging.getLogger(__name__)


def period_minute_df(
    pair: TurbinePair, *, start: pd.Timestamp, end_excl: pd.Timestamp, fastlog_dirs: list[Path]
) -> pd.DataFrame:
    """Return the FLORIS inputs for every minute of one scoring period, from the upstream turbine's fastlog only.

    ``fit_k`` marks the minutes that pass the scoring filter on the upstream turbine, before any gap is filled.
    """
    wtg = pair.upstream
    fl_df = load_period_fl_df([wtg], start=start, end_excl=end_excl, fastlog_dirs=fastlog_dirs)
    minute_df = make_minute_df(fl_df, free_stream_wtg=wtg, wtgs=[wtg])
    minute_df[f"{wtg}_normal"] = normal_minutes(fl_df[wtg], wtg=wtg)
    minute_df = minute_df.reindex(pd.date_range(start, end_excl, freq="1min", inclusive="left"))
    minute_df["fit_k"] = (
        minute_df[FREE_STREAM_COLS].notna().all(axis=1)
        & minute_df[f"{wtg}_yaw_misalignment"].notna()
        & minute_df["wd"].between(*pair.wd_range_deg, inclusive="left")
        & minute_df["ws_raw"].between(*WS_RANGE_MS, inclusive="left")
        & minute_df[f"{wtg}_normal"].fillna(value=False).astype(bool)
        & minute_df[f"{wtg}_power"].gt(MIN_UPSTREAM_POWER_KW)
    )
    n_filled = int(minute_df[FREE_STREAM_COLS].isna().any(axis=1).sum())
    n_no_yaw = int(minute_df[f"{wtg}_yaw_misalignment"].isna().sum())
    if n_filled or n_no_yaw:
        logger.info(
            "%s %s: %d minutes with a filled free-stream input, %d with yaw misalignment set to zero",
            pair.name,
            start,
            n_filled,
            n_no_yaw,
        )
    # the direction does not cross north in any scoring period, so a plain fill is safe
    minute_df[FREE_STREAM_COLS] = minute_df[FREE_STREAM_COLS].ffill().bfill()
    minute_df[f"{wtg}_yaw_misalignment"] = minute_df[f"{wtg}_yaw_misalignment"].fillna(0.0)
    if minute_df[FREE_STREAM_COLS].isna().any(axis=None):
        msg = f"no usable {wtg} free-stream inputs from {start} to {end_excl}"
        raise ValueError(msg)
    minute_df[PAIR_COL] = pair.name
    return minute_df


def fit_k(*, minute_df: pd.DataFrame, wtg: str, fmodel: FlorisModel, turbine_names: list[str]) -> float:
    """Return the zero-bias wind speed scale for ``wtg``'s own power on the ``fit_k`` minutes of ``minute_df``."""
    fit_df = minute_df.loc[minute_df["fit_k"]]
    yaw = yaw_angles_for(df=fit_df, turbine_names=turbine_names, wtg=wtg)
    k = fit_ws_scale(
        modelled_power_fn=lambda k: run_floris_minutes(
            fmodel, turbine_names=turbine_names, df=fit_df, k=k, yaw_angles=yaw
        )[wtg].to_numpy(),
        measured_power=fit_df[f"{wtg}_power"],
    )
    logger.info("%s: fitted k=%.4f on %d minutes", wtg, k, len(fit_df))
    return k


def make_example_submission(*, fastlog_dirs: list[Path], metadata_df: pd.DataFrame) -> pd.DataFrame:
    """Return a submission, indexed and ordered like ``scoring.submission_template()``."""
    fmodel, turbine_names = load_hot_floris_model(metadata_df)
    submission = submission_template()
    for wtg in UPSTREAM_WTGS:
        pairs = [pair for pair in PAIRS if pair.upstream == wtg]
        minute_dfs = {
            (pair.name, start): period_minute_df(pair, start=start, end_excl=end_excl, fastlog_dirs=fastlog_dirs)
            for pair in pairs
            for start, end_excl in pair.periods
        }
        k = fit_k(minute_df=pd.concat(minute_dfs.values()), wtg=wtg, fmodel=fmodel, turbine_names=turbine_names)
        for pair in pairs:
            df = pd.concat([minute_df for (name, _), minute_df in minute_dfs.items() if name == pair.name])
            yaw = yaw_angles_for(df=df, turbine_names=turbine_names, wtg=wtg)
            power = run_floris_minutes(fmodel, turbine_names=turbine_names, df=df, k=k, yaw_angles=yaw)
            submission.loc[df.index, DOWNSTREAM_POWER_COL] = power[pair.downstream].to_numpy()
            if pair.lidar:
                lidar_df = df.assign(**{f"{pair.downstream}_yaw_position": df["wd"]})
                los = modelled_los_minutes(
                    fmodel, turbine_names=turbine_names, df=lidar_df, k=k, yaw_angles=yaw, lidar_wtg=pair.downstream
                )
                submission.loc[df.index, LEFT_LOS_COL] = los[(LOS_RANGE_M, HUB_HEIGHT_M, "Left")].to_numpy()
                submission.loc[df.index, RIGHT_LOS_COL] = los[(LOS_RANGE_M, HUB_HEIGHT_M, "Right")].to_numpy()
    return submission


def write_submission(submission: pd.DataFrame, path: Path) -> None:
    """Validate ``submission`` and write it in the template's CSV format."""
    out = submission.copy()
    out.index = out.index.strftime(TIMESTAMP_FORMAT).rename(TIMESTAMP_COL)
    validate_submission(out.reset_index())
    logger.info("Writing: %s", path)
    out.to_csv(path, lineterminator="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a FLORIS example submission for the wake steering challenge.")
    parser.add_argument("--fastlog-dir", type=Path, action="append", dest="fastlog_dirs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    setup_logger(out_dir / f"{Path(__file__).stem}.log")
    submission = make_example_submission(
        fastlog_dirs=args.fastlog_dirs or [get_filestore_dir()], metadata_df=unpack_local_meta_data()
    )
    write_submission(submission, args.out or out_dir / "submission.csv")


if __name__ == "__main__":
    main()
