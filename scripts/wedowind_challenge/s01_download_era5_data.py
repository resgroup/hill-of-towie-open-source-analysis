# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "hot-open",
#     "cdsapi",
#     "tqdm",
# ]
#
# [tool.uv.sources]
# hot-open = { path = "../../" }
# ///

"""Download ERA5 data from the Copernicus Climate Change Service (C3S) Climate Data Store (CDS).

Follow the instructions to set up your CDS API key: https://cds.climate.copernicus.eu/how-to-api

This script can be run with uv directly from the root of the repository:
`uv run scripts/wedowind_challenge/s01_download_era5_data.py`
"""

# ruff: noqa: G004

import logging
from collections.abc import Iterable
from pathlib import Path

import cdsapi
from tqdm import tqdm

from hot_open.paths import ANALYSES_DIR

logger = logging.getLogger(__name__)


def generate_retrieve_args(
    variables: list[str], area: list[float], years: Iterable[int], output_dir: Path
) -> list[dict]:
    """Generate arguments for the CDS API to retrieve ERA5 data."""
    all_days = [f"{i:02}" for i in range(1, 31 + 1)]
    all_times = [f"{i:02}:00" for i in range(23 + 1)]

    calls_args = []
    for year in years:
        for month in range(1, 12 + 1):
            args = {
                "name": "reanalysis-era5-single-levels",
                "request": {
                    "product_type": ["reanalysis"],
                    "variable": variables,
                    "year": [str(year)],
                    "month": [f"{month:02}"],
                    "day": all_days,
                    "time": all_times,
                    "data_format": "grib",
                    "download_format": "unarchived",
                    "area": area,
                },
                "target": (output_dir / f"ERA5_{year}_{month}.grib").as_posix(),
            }
            calls_args.append(args)

    return calls_args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    output_dir = ANALYSES_DIR / "wedowind_competition_input_data" / ".era5"
    output_dir.mkdir(exist_ok=True, parents=True)

    force_redownload = False
    request_args = generate_retrieve_args(
        variables=[
            "2m_temperature",
            "100m_u_component_of_wind",
            "100m_v_component_of_wind",
        ],
        area=[57.75, -3.25, 57.5, -3],
        years=range(2016, 2020 + 1),
        output_dir=output_dir,
    )

    client = cdsapi.Client()
    for call_kwargs in tqdm(request_args):
        if not force_redownload and Path(call_kwargs["target"]).is_file():
            logger.info(f"File {call_kwargs['target']} already cached")
            continue
        logger.info(f"Generating data: {call_kwargs['target']}")
        logger.debug(f"Requesting data with: {call_kwargs}")
        client.retrieve(**call_kwargs)
