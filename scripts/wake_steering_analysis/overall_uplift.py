import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hot_open.era5_helpers import get_hot_era5_hourly_df, HOT_LAT, HOT_LON
from wind_up.caching import with_parquet_cache
from wind_up.combine_results import combine_results
from wind_up.constants import TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import MastOrLiDARDataset, ReanalysisDataset

from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_10min_data
from hot_open.settings import get_cache_dir, get_out_dir, get_wind_up_output_dir
from hot_open.unpack import unpack_local_meta_data, unpack_local_scada_data_v2
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.inspect_data import LOCAL_TEMPORARY_DIR

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "wind_up_config"


def _hot_initialize_toggle(*, initial_time: pd.Timestamp, toggle_half_period_minutes: int) -> tuple[int, float]:
    """Establish the initial toggle state and timer for a given timestamp.

    :param initial_time: given timestamp for which to determine the toggle state
    :param toggle_half_period_minutes: half period of the toggle in minutes
    :return: the correct toggle state and toggle timer given the `initial_time` and toggle period.
    """
    now_unix = initial_time.timestamp()
    toggle_half_period_seconds = toggle_half_period_minutes * 60
    toggle_state = 1 if int(now_unix // toggle_half_period_seconds) % 2 == 0 else 0
    toggle_start_time = pd.Timestamp(now_unix - (now_unix % toggle_half_period_seconds), unit="s", tz="UTC")
    toggle_timer_s = (initial_time - toggle_start_time).total_seconds()
    return toggle_state, toggle_timer_s


def _hot_dy_toggle_series(*, timestamps: pd.DatetimeIndex, toggle_half_period_minutes: int) -> pd.Series:
    """Generate a series of HoT toggle states for given timestamps.

    :param timestamps: timestamps for which to generate toggle states
    :param toggle_half_period_minutes: half period of the toggle in minutes
    :return: series of toggle states (0 or 1) indexed by the given timestamps
    """
    states: list[int] = []

    for timestamp in timestamps:
        state, _ = _hot_initialize_toggle(initial_time=timestamp, toggle_half_period_minutes=toggle_half_period_minutes)
        states.append(state)

    return pd.Series(states, index=timestamps)


def hot_dy_toggle_df(scada_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with bool cols toggle_on and toggle_off, indexed by TimeStamp_StartFormat."""
    assert scada_df.index.name == "TimeStamp_StartFormat"
    toggle_series = _hot_dy_toggle_series(
        timestamps=pd.DatetimeIndex(
            data=pd.date_range(
                start=scada_df.index.min(),
                end=scada_df.index.max(),
                freq="10min",
                tz="UTC",  # HOT SCADA is UTC End format
                name=TIMESTAMP_COL,
            )
        ),
        toggle_half_period_minutes=50,
    )
    toggle_ok_first_dt = pd.Timestamp("2025-11-05 16:30", tz="UTC")
    return pd.DataFrame(
        data={
            "toggle_on": toggle_series.eq(1) & (toggle_series.index >= toggle_ok_first_dt),
            "toggle_off": toggle_series.eq(0) & (toggle_series.index >= toggle_ok_first_dt),
        },
        index=toggle_series.index,
    )


def _hot_dy_lidar_datasets(data_dir: Path, start_dt, end_dt_excl) -> list[MastOrLiDARDataset]:
    """Load HoT DY Lidar datasets.

    :return: list of MastOrLiDARDataset containing the HoT Lidar data.
    """
    lidar_datasets = []

    lidar_unit_id = "2428"
    lidar_model = "ZX300"
    lidar_datasets.append(
        MastOrLiDARDataset(
            id=f"{lidar_model}_{lidar_unit_id}",
            data=load_zx_lidar_10min_data(
                data_dir=data_dir,
                lidar_unit_id=lidar_unit_id,
                start_dt=start_dt,
                end_dt_excl=end_dt_excl,
                remove_bad_values=True,
            ),
        )
    )

    lidar_unit_id = "5060"
    lidar_model = "ZTM"
    df_5060 = load_zx_lidar_10min_data(
        data_dir=data_dir,
        lidar_unit_id=lidar_unit_id,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )
    df_5060["Met Compass Bearing (deg)"] = (
        df_5060["Met Compass Bearing (deg)"] - 131.5
    ) % 360  # TODO fix in wind-up instead
    lidar_datasets.append(
        MastOrLiDARDataset(
            id=f"{lidar_model}_{lidar_unit_id}",
            data=df_5060,
        )
    )

    return lidar_datasets


def _check_fl_scada_power(scada_df: pd.DataFrame, out_dir: Path | None) -> None:
    scada_power_col = DataColumns.active_power_mean
    fl_power_col = "mean_act_power_fl"
    valid = scada_df[[scada_power_col, fl_power_col]].dropna()
    valid = valid[valid[scada_power_col] > 10]
    corr = valid[scada_power_col].corr(valid[fl_power_col])
    ratio = (valid[fl_power_col] / valid[scada_power_col]).mean()
    logger.info("FL vs SCADA power check: corr=%.5f, FL/SCADA ratio=%.4f (n=%d)", corr, ratio, len(valid))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(valid[scada_power_col], valid[fl_power_col], alpha=0.2, s=1, rasterized=True)
    max_val = max(valid[scada_power_col].max(), valid[fl_power_col].max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=1, label="1:1")
    ax.grid()
    ax.set_xlabel("SCADA ActivePowerMean (kW)")
    ax.set_ylabel("FL ActPower_Value 10min mean (kW)")
    ax.set_title(f"FL vs SCADA power\ncorr={corr:.5f}, FL/SCADA ratio={ratio:.4f}")
    ax.legend()
    if out_dir is not None:
        fig.savefig(out_dir / "fl_vs_scada_power_check.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if corr < 0.999 or not (0.98 <= ratio <= 1.02):
        msg = f"FL vs SCADA power mismatch: {corr=:.5f}, {ratio=:.4f}"
        raise ValueError(msg)


@with_parquet_cache(get_cache_dir() / "unpack_scada_data" / "hot_dy_scada_df.parquet")
def hot_dy_scada_df() -> pd.DataFrame:
    dy_toggle_col = "computed_driver_post_processed_toggle_state"
    dy_wake_steer_col = "computed_core_post_processed_core_wake_steering_offset_degrees"
    dy_yawtarget_col = "computed_driver_post_processed_yaw_target_degrees"

    fl_data_dir = LOCAL_TEMPORARY_DIR / "turbine_fastlog" / "Filestore"
    fl_start_dt = pd.Timestamp("2026-01-01 00:00", tz="UTC")
    fl_end_dt_excl = pd.Timestamp("2026-05-01 00:00", tz="UTC")

    per_turbine_10min: list[pd.DataFrame] = []
    for wtg_number in range(1, 21 + 1):
        logger.info("loading FL data for T%02d (%d/21)", wtg_number, wtg_number)
        wtg_fl_df = load_hot_fl_data(
            data_dir=fl_data_dir,
            wtg_numbers=[wtg_number],
            start_dt=fl_start_dt,
            end_dt_excl=fl_end_dt_excl,
            extra_tags=[dy_toggle_col, dy_yawtarget_col, dy_wake_steer_col],
        )
        turbine_df = wtg_fl_df.droplevel(0, axis=1)
        if "ActPower_Value" not in turbine_df.columns:
            turbine_df["ActPower_Value"] = np.nan
        turbine_10min = (
            turbine_df[[dy_toggle_col, dy_wake_steer_col]]
            .abs()
            .resample("10min")
            .mean()
            .rename(columns={dy_toggle_col: "mean_toggle_state", dy_wake_steer_col: "mean_abs_wake_steer_command"})
            .join(
                turbine_df[[dy_yawtarget_col]]
                .resample("10min")
                .count()
                .rename(columns={dy_yawtarget_col: "count_yaw_target_command_active"}),
                how="outer",
            )
            .join(
                turbine_df[["ActPower_Value"]]
                .resample("10min")
                .mean()
                .rename(columns={"ActPower_Value": "mean_act_power_fl"}),
                how="outer",
            )
            .assign(TurbineName=f"T{wtg_number:02d}")
        )
        del wtg_fl_df, turbine_df
        per_turbine_10min.append(turbine_10min)

    fl_10min_df = pd.concat(per_turbine_10min)
    fl_10min_df.index.name = "TimeStamp_StartFormat"
    fl_stacked = fl_10min_df.set_index("TurbineName", append=True)
    scada_df = unpack_local_scada_data_v2(data_dir=LOCAL_TEMPORARY_DIR)
    scada_df = (
        scada_df.set_index("TurbineName", append=True).join(fl_stacked, how="left").reset_index(level="TurbineName")
    )
    return scada_df


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    # whole farm analysis
    config_file_name = "HOT_dynamic_yaw.yaml"
    save_plots = True
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    cfg.out_dir = get_wind_up_output_dir(cfg.assessment_name)
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")
    cfg.bootstrap_runs_override = 400 // 4  # TODO(AlexClerc): remove

    scada_df = hot_dy_scada_df()
    _check_fl_scada_power(scada_df, out_dir=out_dir)

    metadata_df = unpack_local_meta_data(data_dir=LOCAL_TEMPORARY_DIR, scada_index_name=scada_df.index.name)
    reanalysis_datasets = [
        ReanalysisDataset(
            id=f"ERA5_{HOT_LAT:.2f}_{HOT_LON:.2f}",
            data=get_hot_era5_hourly_df(),
        )
    ]
    toggle_df = hot_dy_toggle_df(scada_df)

    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df,
        metadata_df=metadata_df,
        toggle_df=toggle_df,
        reanalysis_datasets=reanalysis_datasets,
        cache_dir=get_cache_dir() / "windup_cache" / cfg.assessment_name,
        mast_or_lidar_datasets=_hot_dy_lidar_datasets(
            data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
            start_dt=scada_df.index.min(),
            end_dt_excl=scada_df.index.max(),
        ),
    )
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
    combined_results_df = combine_results(results_per_test_ref_df, plot_config=plot_cfg, auto_choose_refs=False)
    combined_results_df.to_csv(
        cfg.out_dir / f"{cfg.assessment_name}_combined_results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    # TODO visualize results eg as bubble plot
