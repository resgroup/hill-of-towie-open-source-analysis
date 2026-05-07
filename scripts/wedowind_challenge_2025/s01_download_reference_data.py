"""Download ERA5 reference data via Open-Meteo API and save to output directory."""

import logging
from pathlib import Path

from hot_open.era5_helpers import HOT_LAT, HOT_LON, get_hot_era5_hourly_df
from hot_open.settings import get_out_dir
from scripts.logger import setup_logger

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    logger.info("log file is at %s", log_path)

    hourly_dataframe = get_hot_era5_hourly_df()

    fpath = out_dir / f"ERA5_{HOT_LAT:.2f}_{HOT_LON:.2f}.parquet"
    hourly_dataframe.to_parquet(fpath)
    logger.info("saved ERA5 data to %s", fpath)
