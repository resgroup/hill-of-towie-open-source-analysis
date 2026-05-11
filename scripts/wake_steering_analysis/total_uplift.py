import logging
from pathlib import Path

from hot_open.settings import get_out_dir
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.uplift_no_steering import run_uplift_no_steering
from scripts.wake_steering_analysis.uplift_per_steer import run_uplift_per_steer

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    run_uplift_per_steer()
    run_uplift_no_steering()
