import logging
from pathlib import Path

from hot_open.settings import get_out_dir
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.uplift_no_steering import hot_dy_uplift_no_steering
from scripts.wake_steering_analysis.uplift_per_steer import hot_dy_uplift_per_steer

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    rerun_windup = True
    ws_uplift, ws_uplift_uncertainty, ws_steering_turbine_yaph_change = hot_dy_uplift_per_steer(
        rerun_windup=rerun_windup
    )
    cc_uplift, cc_uplift_uncertainty, cc_yaph_change = hot_dy_uplift_no_steering(rerun_windup=rerun_windup)

    # results from 20 year FLORIS simulation with long-term steering config
    lt_steering_time_frac = (
        8.9201 / 100
    )  # long-term fraction of time steering turbines will spend in their steering directions with long-term steering config
    lt_most_steering_turbine = "T03"
    lt_most_steering_turbine_steering_time_frac = 34.250 / 100
    lt_steering_mwh_frac = (
        18.3768 / 100
    )  # long-term fraction of MWh for steering + dependent turbine pairs in their steering directions
    ws_uplift_lt_adjustment = 0.958824  # adjusts campaign uplift result to long-term wind speed and day/night split

    lt_yaph_change = (
        lt_steering_time_frac * ws_steering_turbine_yaph_change + (1 - lt_steering_time_frac) * cc_yaph_change
    )
    lt_yaph_change_per_turbine_max = (
        lt_most_steering_turbine_steering_time_frac * ws_steering_turbine_yaph_change
        + (1 - lt_most_steering_turbine_steering_time_frac) * cc_yaph_change
    )
    lt_uplift_p50 = lt_steering_mwh_frac * ws_uplift_lt_adjustment * ws_uplift + (1 - lt_steering_mwh_frac) * cc_uplift
    lt_uplift_uncertainty = (
        (lt_steering_mwh_frac * ws_uplift_lt_adjustment * ws_uplift_uncertainty) ** 2
        + ((1 - lt_steering_mwh_frac) * cc_uplift_uncertainty) ** 2
    ) ** 0.5
    lt_uplift_p95 = lt_uplift_p50 - 1.645 * lt_uplift_uncertainty

    msg = f"AEP uplift P50 {100 * lt_uplift_p50:.1f}%, uncertainty {100 * lt_uplift_uncertainty:.1f}%, P95 {100 * lt_uplift_p95:.1f}%"
    logger.info(msg)
    msg = f"Long term yaw activity change {100 * lt_yaph_change:.1f}% (wind farm average), individual turbine maximum +{100 * lt_yaph_change_per_turbine_max:.1f}% ({lt_most_steering_turbine})"
    logger.info(msg)
