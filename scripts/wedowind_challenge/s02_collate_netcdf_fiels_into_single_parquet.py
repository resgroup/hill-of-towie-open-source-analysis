# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "h5netcdf",
#     "hot-open",
#     "xarray",
# ]
#
# [tool.uv.sources]
# hot-open = { path = "../../" }
# ///

"""Convert ERA5 netcdf data to parquet data.

Data Documentation: https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation

This script can be run with uv directly from the root of the repository:
`uv run scripts/wedowind_challenge/s02_collate_netcdf_fiels_into_single_parquet.py`
"""

# ruff: noqa: PD901,G004

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr

from hot_open.paths import ANALYSES_DIR

logger = logging.getLogger(__name__)


def _convert_u_v_to_ws_wd(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ws = np.sqrt(u**2 + v**2)
    wd = np.mod(180 / np.pi * np.atan2(u, v), 360)
    return ws, wd


def _convert_temperature_to_celsius(t: np.ndarray) -> np.ndarray:
    return t - 273.15


if __name__ == "__main__":
    chosen_dir = ANALYSES_DIR / "wedowind_competition_input_data" / ".era5"

    location_data = defaultdict(list)
    for file_path in chosen_dir.glob("*.nc"):
        ds = xr.open_dataset(file_path)
        df = ds.to_dataframe()

        for (lat, lon), sub_df in df.groupby(["latitude", "longitude"]):
            ws, wd = _convert_u_v_to_ws_wd(u=sub_df["u100"].to_numpy(), v=sub_df["v100"].to_numpy())
            _df = pd.DataFrame(
                {
                    "ERA5_wind_speed_m/s": ws,
                    "ERA5_wind_direction_°": wd,
                    "ERA5_temperature_at_10m_": _convert_temperature_to_celsius(sub_df["t2m"].to_numpy()),
                },
                index=sub_df.index.get_level_values("valid_time")  # type: ignore[attr-defined]
                .astype("datetime64[us]")
                .tz_localize("UTC")
                .rename("TimeStamp_StartFormat"),
            )
            location_data[(lat, lon)].append(_df)

        for location, dfs in location_data.items():
            lat, lon = location
            fpath = chosen_dir / f"ERA5_{lat:.2f}_{lon:.2f}.parquet"
            pd.concat(dfs).to_parquet(fpath)
