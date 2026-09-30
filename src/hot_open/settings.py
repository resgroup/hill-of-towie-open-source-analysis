"""Settings and path configuration for the hill-of-towie-open-source-analysis package."""

import logging
import os
from functools import cache
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parents[2]
REPO_NAME = REPO_ROOT.stem


@cache
def _load_env() -> None:
    """Populate os.environ from .env files, once per process.

    Precedence, highest first: the real environment, ``HOT_OPEN_DOTENV``, the nearest .env above
    the current working directory (the application's), then ``REPO_ROOT/.env``. The last is an
    explicit path because ``find_dotenv()`` falls back to the cwd when ``__main__`` has no
    ``__file__`` (REPL, Jupyter). Tests that write a .env must call ``_load_env.cache_clear()``.
    """
    explicit = os.getenv("HOT_OPEN_DOTENV")
    if explicit:
        load_dotenv(explicit, override=False)
    cwd_dotenv = find_dotenv(usecwd=True)
    if cwd_dotenv:
        load_dotenv(cwd_dotenv, override=False)
    repo_dotenv = REPO_ROOT / ".env"
    if repo_dotenv.is_file():
        load_dotenv(repo_dotenv, override=False)


def get_data_dir(*, log_message: bool = False) -> Path:
    """Get the HOT open data directory where all input data files should be.

    Can be customized by setting the "HOT_OPEN_DATA_DIR" enviroment variable.
    """
    _load_env()
    path = Path(os.getenv("HOT_OPEN_DATA_DIR", Path.home() / "temp" / REPO_NAME / "data"))
    if log_message:
        msg = f"Data directory is {path}"
        logger.info(msg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_out_dir(*, dir_name: str, subdir_name: str | None = None, subsubdir_name: str | None = None) -> Path:
    """Get the output directory.

    Can be customized by setting the "HOT_OPEN_OUTPUT_DIR" enviroment variable.
    """
    _load_env()
    path = Path(os.getenv("HOT_OPEN_OUTPUT_DIR", Path.home() / "temp" / REPO_NAME / "output")) / dir_name
    path = path / subdir_name if subdir_name else path
    path = path / subsubdir_name if subsubdir_name else path
    msg = f"Output directory is {path}"
    logger.info(msg)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_cache_dir(*, log_message: bool = False) -> Path:
    """Get the cache directory where cached intermediate files should be.

    Can be customized by setting the "HOT_OPEN_CACHE_DIR" enviroment variable.
    """
    _load_env()
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
    _load_env()
    location = Path(os.getenv("HOT_OPEN_FILESTORE_DIR", get_data_dir() / "turbine_fastlog" / "Filestore"))
    location.mkdir(exist_ok=True, parents=True)
    return location


def get_wind_up_output_dir(analysis_name: str = "hill-of-towie-open-source-analysis") -> Path:
    """Get the location where wind-up output will be saved.

    Defaulted to: `[user folder]/.windup/analyses/[analysis_name]`

    But can customized by setting the "WINDUP_OUTPUT_DIR" enviroment variable, in
    which case the location will be: `[WINDUP_OUTPUT_DIR]/[analysis_name]`
    """
    _load_env()
    location = Path(os.getenv("WINDUP_OUTPUT_DIR", Path.home() / "temp" / REPO_NAME / "windup_output"))
    analysis_directory = location / analysis_name
    analysis_directory.mkdir(exist_ok=True, parents=True)
    return analysis_directory
