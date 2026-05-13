import pandas as pd
from wake_steering_analysis.inspect_data import LOCAL_TEMPORARY_DIR

from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_fl_data
from hot_open.settings import get_filestore_dir

if __name__ == "__main__":
    start_dt = pd.Timestamp("2026-01-14 00:00:00", tz="UTC")
    end_dt_excl = pd.Timestamp("2026-01-14 12:00:00", tz="UTC")

    zx300_fl_df = load_zx_lidar_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="2428",
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )

    # TODO populate helpful columns shear, veer, TI columns in zx300_fl_df

    wtg_numbers = [11, 14]

    dy_toggle_col = "computed_driver_post_processed_toggle_state"
    dy_wake_steer_col = "computed_core_post_processed_core_wake_steering_offset_degrees"
    dy_wd_col = "computed_core_post_processed_consensus_wind_direction_true_degrees"
    dy_northed_yawpos_col = "computed_driver_pre_processed_yaw_direction_true_degrees"
    dy_yawtarget_col = "computed_driver_post_processed_yaw_target_degrees"

    wtg_fl_df = load_hot_fl_data(
        data_dir=get_filestore_dir(),
        wtg_numbers=wtg_numbers,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        extra_tags=[dy_toggle_col, dy_northed_yawpos_col, dy_yawtarget_col, dy_wd_col, dy_wake_steer_col],
    )

    # make plots
