import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parents[2]
REPO_NAME = REPO_ROOT.stem


def get_out_dir(*, dir_name: str, subdir_name: str | None = None, subsubdir_name: str | None = None) -> Path:
    """Get the output directory.

    Can be customized by setting the "HOT_OPEN_OUTPUT_DIR" enviroment variable.
    """
    load_dotenv()
    path = Path(os.getenv("HOT_OPEN_OUTPUT_DIR", Path.home() / "temp" / REPO_NAME / "output")) / dir_name
    path = path / subdir_name if subdir_name else path
    path = path / subsubdir_name if subsubdir_name else path
    msg = f"Output directory is {path}"
    logger.info(msg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir(*, log_message: bool = False) -> Path:
    """Get the cache directory where input parquet files should be.

    Can be customized by setting the "HOT_OPEN_CACHE_DIR" enviroment variable.
    """
    load_dotenv()
    path = Path(os.getenv("HOT_OPEN_CACHE_DIR", Path.home() / "temp" / REPO_NAME / "cache"))
    if log_message:
        msg = f"Cache directory is {path}"
        logger.info(msg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_filestore_dir() -> Path:
    r"""Get the local Filestore directory.

    Can customized by setting the "HOT_OPEN_FILESTORE_DIR" enviroment variable.
    """
    load_dotenv()
    location = Path(os.getenv("HOT_OPEN_FILESTORE_DIR", Path.home() / "Filestore"))
    location.mkdir(exist_ok=True, parents=True)
    return location


def get_wind_up_output_dir(analysis_name: str = "hill-of-towie-open-source-analysis") -> Path:
    """Get the location where wind-up output will be saved.

    Defaulted to: `[user folder]/.windup/analyses/[analysis_name]`

    But can customized by setting the "WINDUP_OUTPUT_DIR" enviroment variable, in
    which case the location will be: `[WINDUP_OUTPUT_DIR]/[analysis_name]`
    """
    load_dotenv()
    location = Path(os.getenv("WINDUP_OUTPUT_DIR", Path.home() / ".windup" / "analyses"))
    analysis_directory = location / analysis_name
    analysis_directory.mkdir(exist_ok=True, parents=True)
    return analysis_directory
