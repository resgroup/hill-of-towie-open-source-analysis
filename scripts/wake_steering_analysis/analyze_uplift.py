import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from wind_up.constants import TIMESTAMP_COL
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import MastOrLiDARDataset, ReanalysisDataset

from hot_open.lidar_helpers import load_zx_lidar_10min_data
from hot_open.settings import get_cache_dir, get_out_dir, get_wind_up_output_dir
from hot_open.unpack import unpack_local_meta_data, unpack_local_scada_data_v2
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.inspect_data import LOCAL_TEMPORARY_DIR

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "wind_up_config"


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


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    scada_df = unpack_local_scada_data_v2(data_dir=LOCAL_TEMPORARY_DIR)
    metadata_df = unpack_local_meta_data(data_dir=LOCAL_TEMPORARY_DIR, scada_index_name=scada_df.index.name)
    hot_best_era5 = "ERA5T_57.50N_-3.25E_100m_1hr"
    reanalysis_datasets = [
        ReanalysisDataset(
            id=hot_best_era5,
            data=pd.read_parquet(Path(__name__).parent / "reanalysis_data" / f"{hot_best_era5}_20260331.parquet"),
        )
    ]
    toggle_df = hot_dy_toggle_df(scada_df)

    # analysis is performed by wake steer
    wakesteer_table = pd.read_csv(Path(__file__).parent / "controller_config" / "wake-steering-lookup.csv")
    wake_steers = _extract_hot_wake_steers_from_table(wakesteer_table)
    config_file_name = "HoT_dynamic_yaw.yaml"
    save_plots = True
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    cfg.out_dir = get_wind_up_output_dir(cfg.assessment_name)
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")
    cfg.bootstrap_runs_override = 400 // 4  # TODO(AlexClerc): remove

    all_wakesteer_results = []
    pd.DataFrame([asdict(x) for x in wake_steers]).to_csv(cfg.out_dir / "wake_steer_meta.csv", index=False)

    for wakesteer in wake_steers:
        msg = f"{wakesteer.upwind_wtg} -> {wakesteer.downwind_wtg}"
        logger.info(msg)
        wakesteer_cfg = cfg.model_copy()
        wakesteer_cfg.test_wtgs = [
            x.model_copy() for x in cfg.asset.wtgs if x.name in [wakesteer.upwind_wtg, wakesteer.downwind_wtg]
        ]
        direction_margin = 0  # TODO need to determine this
        wakesteer_cfg.ref_wd_filter = [
            (wakesteer.first_wdir - direction_margin) % 360,
            (wakesteer.last_wdir + direction_margin) % 360,
        ]
        assessment_inputs = AssessmentInputs.from_cfg(
            cfg=wakesteer_cfg,
            plot_cfg=plot_cfg,
            scada_df=scada_df,
            metadata_df=metadata_df,
            toggle_df=toggle_df,
            reanalysis_datasets=reanalysis_datasets,
            cache_dir=get_cache_dir() / cfg.assessment_name,
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
