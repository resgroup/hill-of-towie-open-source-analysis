import itertools
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wind_up.caching import with_parquet_cache
from wind_up.combine_results import calc_net_uplift
from wind_up.constants import TIMESTAMP_COL, DataColumns
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.scada_funcs_plots import bubble_plot
from wind_up.reanalysis_data import MastOrLiDARDataset
from wind_up.smart_data import add_smart_lat_long_to_cfg

from hot_open.era5_helpers import get_hot_reanalysis_datasets
from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_10min_data
from hot_open.settings import get_cache_dir, get_data_dir, get_filestore_dir, get_out_dir, get_wind_up_output_dir
from hot_open.unpack import unpack_local_meta_data, unpack_local_scada_data_v2
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.combine_uplift_per_steer import (
    combine_wakesteer_results_with_yaw,
)
from scripts.wake_steering_analysis.hot_wake_steering_helpers import CONFIG_DIR

logger = logging.getLogger(__name__)


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

    fl_data_dir = get_filestore_dir()
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
    scada_df = unpack_local_scada_data_v2(data_dir=get_data_dir())
    scada_df = (
        scada_df.set_index("TurbineName", append=True).join(fl_stacked, how="left").reset_index(level="TurbineName")
    )
    return scada_df


@dataclass
class HOTWakeSteer:
    upwind_wtg: str
    downwind_wtg: str
    first_wdir: float
    last_wdir: float
    wake_centre: float = np.nan


def _extract_hot_wake_steers_from_table(wakesteer_table: pd.DataFrame) -> list[HOTWakeSteer]:
    wakesteer_table = wakesteer_table.copy()
    wakesteer_table = wakesteer_table.rename(
        columns={
            "device_name": "TurbineName",
            "wind_direction_degrees": "WindDirection_deg",
            "yaw_offset_degrees": "YawOffset_deg",
            "dependent_device_name": "DependentTurbineNames",
        }
    )
    draft_wake_steers = []
    found_a_steer = False
    wdir_from_previous_row = 0
    for row in wakesteer_table.itertuples():
        if abs(row.YawOffset_deg) > 0 and not found_a_steer:
            found_a_steer = True
            first_wdir = row.WindDirection_deg
            upwind_wtg = row.TurbineName
            downwind_wtg = row.DependentTurbineNames
        if row.YawOffset_deg == 0 and found_a_steer:
            found_a_steer = False
            last_wdir = wdir_from_previous_row
            draft_wake_steers.append(
                HOTWakeSteer(
                    upwind_wtg=upwind_wtg, downwind_wtg=downwind_wtg, first_wdir=first_wdir, last_wdir=last_wdir
                )
            )
        wdir_from_previous_row = row.WindDirection_deg
    # combine wake steers that have crossed 360
    wake_steers = []
    redundant_steers = []
    for wake_steer in draft_wake_steers:
        if wake_steer.first_wdir == 0:
            # find the other wake steer with same upwind and downwind wtgs
            for other_wake_steer in reversed(draft_wake_steers):
                if (
                    other_wake_steer.upwind_wtg == wake_steer.upwind_wtg
                    and other_wake_steer.downwind_wtg == wake_steer.downwind_wtg
                    and other_wake_steer.first_wdir > wake_steer.last_wdir
                ):
                    wake_steers.append(
                        HOTWakeSteer(
                            upwind_wtg=wake_steer.upwind_wtg,
                            downwind_wtg=wake_steer.downwind_wtg,
                            first_wdir=other_wake_steer.first_wdir,
                            last_wdir=wake_steer.last_wdir,
                        )
                    )
                    redundant_steers.append(other_wake_steer)
                    break
        elif wake_steer not in redundant_steers:
            wake_steers.append(wake_steer)

    # populate wake_centre
    for wake_steer in wake_steers:
        wake_steer_df = wakesteer_table[
            (wakesteer_table["TurbineName"] == wake_steer.upwind_wtg)
            & (wakesteer_table["DependentTurbineNames"] == wake_steer.downwind_wtg)
        ]
        idxmax = wake_steer_df["YawOffset_deg"].diff().idxmax()
        if wake_steer_df["YawOffset_deg"].diff().loc[idxmax] > 20:
            wake_steer.wake_centre = wake_steer_df.loc[idxmax, "WindDirection_deg"] - 0.5
    return wake_steers


def _post_process_ws_results(ws_combined_results: pd.DataFrame) -> tuple[float, float, float]:
    min_ws_hours = 100
    ws_combined_results_filt = ws_combined_results[ws_combined_results["hours_on"] >= min_ws_hours]

    ws_uplift = ws_combined_results_filt["p50_net_uplift"].mean()
    ws_uplift_uncertainty = (1 / len(ws_combined_results_filt["sigma"])) * (
        (ws_combined_results_filt["sigma"] ** 2).sum() ** 0.5
    )
    ws_steering_turbine_yaph_change = ws_combined_results_filt["yaph_change"].mean()
    msg = f"WS results P50 {100 * ws_uplift:.2f}%, uncertainty {100 * ws_uplift_uncertainty:.2f}%"
    logger.info(msg)
    msg = f"WS test results yaw activity change for steering turbines: {100 * ws_steering_turbine_yaph_change:.1f}%"
    logger.info(msg)
    return ws_uplift, ws_uplift_uncertainty, ws_steering_turbine_yaph_change


def hot_dy_uplift_per_steer(rerun_windup: bool = True) -> tuple[float, float, float]:
    config_file_name = "HOT_dynamic_yaw.yaml"
    save_plots = True
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    cfg.out_dir = get_wind_up_output_dir(cfg.assessment_name)
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")

    if not rerun_windup:
        ws_combined_results = pd.read_csv(cfg.out_dir / "uplift_per_steer_combined_results_with_yaw.csv")
        ws_uplift, ws_uplift_uncertainty, ws_steering_turbine_yaph_change = _post_process_ws_results(
            ws_combined_results
        )
        return ws_uplift, ws_uplift_uncertainty, ws_steering_turbine_yaph_change

    scada_df = hot_dy_scada_df()

    wakesteer_table = pd.read_csv(Path(__file__).parent / "controller_config" / "wake-steering-lookup.csv")
    wake_steers = _extract_hot_wake_steers_from_table(wakesteer_table)
    all_wakesteer_results = []
    pd.DataFrame([asdict(x) for x in wake_steers]).to_csv(cfg.out_dir / "wake_steer_meta.csv", index=False)
    lidar_refs = ["ZX300_2428"]
    refs = [x.name for x in cfg.ref_wtgs] + cfg.non_wtg_ref_names
    for wakesteer, ref_name in itertools.product(wake_steers, refs):
        msg = f"{wakesteer.upwind_wtg} -> {wakesteer.downwind_wtg} with ref {ref_name}"
        logger.info(msg)

        wakesteer_cfg = cfg.model_copy()
        wakesteer_cfg.use_lt_distribution = (
            False  # long term extrapolation is complex for wake steering and needs to be done outside wind-up
        )
        wakesteer_cfg.test_wtgs = [
            x.model_copy() for x in cfg.asset.wtgs if x.name in [wakesteer.upwind_wtg, wakesteer.downwind_wtg]
        ]
        if ref_name in lidar_refs:
            wakesteer_cfg.non_wtg_ref_names = [ref_name]
            wakesteer_cfg.ref_wtgs = []
        else:
            wakesteer_cfg.non_wtg_ref_names = []
            wakesteer_cfg.ref_wtgs = [x.model_copy() for x in cfg.asset.wtgs if x.name == ref_name]
        direction_margin = 0  # 0 looks great for LiDAR. T1 is a little shifted but still looks OK.
        wakesteer_cfg.ref_wd_filter = [
            (wakesteer.first_wdir - direction_margin) % 360,
            (wakesteer.last_wdir + direction_margin) % 360,
        ]
        wakesteer_cfg.out_dir = cfg.out_dir / f"{wakesteer.upwind_wtg}_{wakesteer.downwind_wtg}_{ref_name}"
        wakesteer_cfg.out_dir.mkdir(parents=True, exist_ok=True)
        plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=wakesteer_cfg.out_dir / "plots")
        assessment_inputs = AssessmentInputs.from_cfg(
            cfg=wakesteer_cfg,
            plot_cfg=plot_cfg,
            scada_df=scada_df,
            metadata_df=unpack_local_meta_data(data_dir=get_data_dir(), scada_index_name=scada_df.index.name),
            toggle_df=hot_dy_toggle_df(scada_df),
            reanalysis_datasets=get_hot_reanalysis_datasets(),
            cache_dir=get_cache_dir() / "windup_cache" / cfg.assessment_name,
            mast_or_lidar_datasets=_hot_dy_lidar_datasets(
                data_dir=get_data_dir() / "lidar_data",
                start_dt=scada_df.index.min(),
                end_dt_excl=scada_df.index.max(),
            ),
        )
        try:
            results_per_test_ref_df = run_wind_up_analysis(inputs=assessment_inputs)
        except Exception as e:  # noqa: BLE001
            print(f"skipping due to exception {e}")
            continue
        if "unc_one_sigma_frc" not in results_per_test_ref_df.columns:
            continue
        if len(results_per_test_ref_df.dropna(subset=["unc_one_sigma_frc"])) < 2:
            continue
        net_p50, net_p95, net_p5 = calc_net_uplift(results_per_test_ref_df, confidence=0.9)
        all_wakesteer_results.append(
            {
                "upwind_wtg": wakesteer.upwind_wtg,
                "downwind_wtg": wakesteer.downwind_wtg,
                "reference": ref_name,
                "hours_off": results_per_test_ref_df["pp_valid_hours_pre"].min(),
                "hours_on": results_per_test_ref_df["pp_valid_hours_post"].min(),
                "net_p50_uplift": net_p50,
                "net_p95_uplift": net_p95,
                "net_p5_uplift": net_p5,
                "upwind_uplift_p50": results_per_test_ref_df.loc[
                    results_per_test_ref_df["test_wtg"] == wakesteer.upwind_wtg, "uplift_frc"
                ].squeeze(),
                "upwind_uplift_unc": results_per_test_ref_df.loc[
                    results_per_test_ref_df["test_wtg"] == wakesteer.upwind_wtg, "unc_one_sigma_frc"
                ].squeeze(),
                "downwind_uplift_p50": results_per_test_ref_df.loc[
                    results_per_test_ref_df["test_wtg"] == wakesteer.downwind_wtg, "uplift_frc"
                ].squeeze(),
                "downwind_uplift_unc": results_per_test_ref_df.loc[
                    results_per_test_ref_df["test_wtg"] == wakesteer.downwind_wtg, "unc_one_sigma_frc"
                ].squeeze(),
                "first_wdir": wakesteer.first_wdir,
                "last_wdir": wakesteer.last_wdir,
            }
        )
        pd.DataFrame(all_wakesteer_results).to_csv(cfg.out_dir / "uplift_per_steer_results_interim.csv", index=False)
    pd.DataFrame(all_wakesteer_results).to_csv(cfg.out_dir / "uplift_per_steer_results.csv", index=False)
    ws_combined_results = combine_wakesteer_results_with_yaw(
        pd.DataFrame(all_wakesteer_results), wind_up_out_dir=cfg.out_dir
    )
    ws_combined_results.to_csv(cfg.out_dir / "uplift_per_steer_combined_results_with_yaw.csv")
    ws_uplift, ws_uplift_uncertainty, ws_steering_turbine_yaph_change = _post_process_ws_results(ws_combined_results)

    min_ws_hours = 100
    ws_combined_results_filt = ws_combined_results[ws_combined_results["hours_on"] >= min_ws_hours]
    del ws_combined_results

    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    cfg.out_dir = get_wind_up_output_dir(cfg.assessment_name)
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")
    (cfg.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    cfg = add_smart_lat_long_to_cfg(md=unpack_local_meta_data(), cfg=cfg)
    west_cfg = cfg.copy()
    west_cfg.asset.wtgs = [x for x in cfg.asset.wtgs if x.name not in [f"T{i:02d}" for i in range(16, 21 + 1)]]
    title = f"{cfg.asset.name} WS upwind turbine mean abs wakesteer"
    bubble_plot(
        cfg=west_cfg,
        series=ws_combined_results_filt.set_index("upwind_wtg")["mean_abs_wakesteer_col_post"],
        title=title,
        cbarunits="deg",
        save_path=plot_cfg.plots_dir / f"{title}.png",
        show_plot=plot_cfg.show_plots,
    )
    upwind_uplifts = ws_combined_results_filt[["upwind_wtg", "upwind_uplift_p50"]].rename(
        columns={"upwind_wtg": "test_wtg", "upwind_uplift_p50": "p50_uplift"}
    )
    downwind_uplifts = ws_combined_results_filt[["downwind_wtg", "downwind_uplift_p50"]].rename(
        columns={"downwind_wtg": "test_wtg", "downwind_uplift_p50": "p50_uplift"}
    )
    title = f"{cfg.asset.name} WS test turbine uplift"
    bubble_plot(
        cfg=west_cfg,
        series=pd.concat([upwind_uplifts, downwind_uplifts]).set_index("test_wtg")["p50_uplift"].sort_index() * 100,
        title=title,
        cbarunits="%",
        save_path=plot_cfg.plots_dir / f"{title}.png",
        show_plot=plot_cfg.show_plots,
    )
    # make a scatter plot of yaw activity change vs wake steer command
    title = f"{cfg.asset.name} yaw activity change vs wakesteer command"
    plt.figure()
    plt.scatter(ws_combined_results_filt["mean_abs_wakesteer_col_post"], 100 * ws_combined_results_filt["yaph_change"])
    plt.xlabel("Mean abs wakesteer command [deg]")
    plt.ylabel("Yaw activity change [%]")
    plt.grid()
    plt.title(title)
    plt.ylim(0, 50)
    plt.xlim(0, 5)
    plt.savefig(plot_cfg.plots_dir / f"{title}.png")
    plt.close()

    return ws_uplift, ws_uplift_uncertainty, ws_steering_turbine_yaph_change


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)
    hot_dy_uplift_per_steer()
