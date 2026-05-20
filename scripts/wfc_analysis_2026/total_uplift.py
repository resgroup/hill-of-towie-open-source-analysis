import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

from hot_open.settings import get_out_dir
from scripts.logger import setup_logger
from scripts.wfc_analysis_2026.uplift_cc import hot_dy_uplift_no_steering
from scripts.wfc_analysis_2026.uplift_ws import hot_dy_uplift_per_steer

logger = logging.getLogger(__name__)


@dataclass
class LTUpliftResult:
    lt_uplift_p50: float
    lt_uplift_uncertainty: float
    lt_uplift_p95: float
    lt_yaph_change: float
    lt_yaph_change_per_turbine_max: float


def compute_lt_uplift(
    *,
    ws_uplift: float,
    ws_uplift_uncertainty: float,
    ws_steering_turbine_yaph_change: float,
    cc_uplift: float,
    cc_uplift_uncertainty: float,
    cc_yaph_change: float,
    lt_steering_time_frac: float = 8.9201 / 100,
    lt_most_steering_turbine_steering_time_frac: float = 34.250 / 100,
    lt_steering_mwh_frac: float = 18.3768 / 100,
    ws_uplift_lt_adjustment: float = 0.958824,
) -> LTUpliftResult:
    """Combine WS and CC uplift estimates into a long-term AEP figure.

    Defaults come from a 20-year FLORIS simulation with the long-term steering config.
    """
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
    return LTUpliftResult(
        lt_uplift_p50=lt_uplift_p50,
        lt_uplift_uncertainty=lt_uplift_uncertainty,
        lt_uplift_p95=lt_uplift_p95,
        lt_yaph_change=lt_yaph_change,
        lt_yaph_change_per_turbine_max=lt_yaph_change_per_turbine_max,
    )


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}_{dt.datetime.now(tz=dt.UTC).strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    rerun_windup = True
    ws_uplift, ws_uplift_uncertainty, ws_steering_turbine_yaph_change = hot_dy_uplift_per_steer(
        rerun_windup=rerun_windup
    )
    cc_uplift, cc_uplift_uncertainty, cc_yaph_change = hot_dy_uplift_no_steering(rerun_windup=rerun_windup)

    lt_most_steering_turbine = "T03"
    lt = compute_lt_uplift(
        ws_uplift=ws_uplift,
        ws_uplift_uncertainty=ws_uplift_uncertainty,
        ws_steering_turbine_yaph_change=ws_steering_turbine_yaph_change,
        cc_uplift=cc_uplift,
        cc_uplift_uncertainty=cc_uplift_uncertainty,
        cc_yaph_change=cc_yaph_change,
    )

    msg = (
        f"AEP uplift P50 {100 * lt.lt_uplift_p50:.1f}%, "
        f"uncertainty {100 * lt.lt_uplift_uncertainty:.1f}%, "
        f"P95 {100 * lt.lt_uplift_p95:.1f}%"
    )
    logger.info(msg)
    msg = (
        f"Long term yaw activity change {100 * lt.lt_yaph_change:.1f}% (wind farm average), "
        f"individual turbine maximum +{100 * lt.lt_yaph_change_per_turbine_max:.1f}% "
        f"({lt_most_steering_turbine})"
    )
    logger.info(msg)
