"""A module that contains the paths used in the analysis."""

import os
from pathlib import Path

from dotenv import load_dotenv


def get_analyses_directory(analysis_name: str = "hill-of-towie-open-source-analysis") -> Path:
    """Get the location where the analysis will be saved.

    Defaulted to: `[user folder]/.windup/analyses/[analysis_name]`

    But can customized by setting the "WINDUP_ANALYSIS_DIR" enviroment variable, in
    which case the location will be: `[WINDUP_ANALYSIS_DIR]/[analysis_name]`
    """
    load_dotenv()
    location = Path(os.getenv("WINDUP_ANALYSIS_DIR", Path.home() / ".windup" / "analyses"))
    analysis_directory = location / analysis_name
    analysis_directory.mkdir(exist_ok=True, parents=True)
    return analysis_directory


REPO_ROOT = Path(__file__).parents[2]

ANALYSES_DIR = get_analyses_directory()
DATA_DIR = ANALYSES_DIR / "zenodo_data"

GLOBAL_CACHE_DIR = ANALYSES_DIR / ".cache"
GLOBAL_CACHE_DIR.mkdir(exist_ok=True, parents=True)
