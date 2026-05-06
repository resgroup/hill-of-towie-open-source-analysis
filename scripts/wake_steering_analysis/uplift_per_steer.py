import itertools
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from wind_up.combine_results import calc_net_uplift
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

from hot_open.settings import get_cache_dir, get_out_dir, get_wind_up_output_dir
from hot_open.unpack import unpack_local_meta_data, unpack_local_scada_data_v2
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.inspect_data import LOCAL_TEMPORARY_DIR
from scripts.wake_steering_analysis.overall_uplift import CONFIG_DIR, _hot_dy_lidar_datasets, hot_dy_toggle_df

logger = logging.getLogger(__name__)


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


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    config_file_name = "HOT_dynamic_yaw.yaml"
    save_plots = True
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    cfg.out_dir = get_wind_up_output_dir(cfg.assessment_name)
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")
    cfg.bootstrap_runs_override = 400 // 4  # TODO(AlexClerc): remove

    scada_df = unpack_local_scada_data_v2(data_dir=LOCAL_TEMPORARY_DIR)
    metadata_df = unpack_local_meta_data(data_dir=LOCAL_TEMPORARY_DIR, scada_index_name=scada_df.index.name)
    hot_best_era5 = "ERA5T_57.50N_-3.25E_100m_1hr"
    reanalysis_datasets = [
        ReanalysisDataset(
            id=hot_best_era5,
            data=pd.read_parquet(Path(__file__).parent / "reanalysis_data" / f"{hot_best_era5}_20260331.parquet"),
        )
    ]
    toggle_df = hot_dy_toggle_df(scada_df)

    wakesteer_table = pd.read_csv(Path(__file__).parent / "controller_config" / "wake-steering-lookup.csv")
    wake_steers = _extract_hot_wake_steers_from_table(wakesteer_table)
    all_wakesteer_results = []
    pd.DataFrame([asdict(x) for x in wake_steers]).to_csv(cfg.out_dir / "wake_steer_meta.csv", index=False)
    lidar_refs = ["ZTM_5060", "ZX300_2428"]
    ok_refs = ["T01", "T13", "T18", "ZX300_2428"]
    for wakesteer, ref_name in itertools.product(wake_steers, ok_refs):
        msg = f"{wakesteer.upwind_wtg} -> {wakesteer.downwind_wtg} with ref {ref_name}"
        logger.info(msg)

        wakesteer_cfg = cfg.model_copy()
        wakesteer_cfg.test_wtgs = [
            x.model_copy() for x in cfg.asset.wtgs if x.name in [wakesteer.upwind_wtg, wakesteer.downwind_wtg]
        ]
        if ref_name in lidar_refs:
            wakesteer_cfg.non_wtg_ref_names = [ref_name]
            wakesteer_cfg.ref_wtgs = []
        else:
            wakesteer_cfg.non_wtg_ref_names = []
            wakesteer_cfg.ref_wtgs = [x.model_copy() for x in cfg.asset.wtgs if x.name == ref_name]
        direction_margin = 0  # TODO need to determine this
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
        try:
            results_per_test_ref_df = run_wind_up_analysis(inputs=assessment_inputs)
        except Exception as e:  # noqa BLE001
            print(f"skipping due to exception {e}")
            continue
        if "unc_one_sigma_frc" not in results_per_test_ref_df.columns:
            continue
        if len(results_per_test_ref_df.dropna(subset=["unc_one_sigma_frc"])) < 2:  # noqa PLR2004
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
