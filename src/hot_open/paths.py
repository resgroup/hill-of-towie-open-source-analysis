"""A module that contains the paths used in the analysis."""
from hot_open.settings import get_wind_up_output_dir

ANALYSES_DIR = get_wind_up_output_dir()
DATA_DIR = ANALYSES_DIR / "zenodo_data"
