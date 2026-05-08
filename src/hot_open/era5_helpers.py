"""ERA5 reanalysis data fetching via Open-Meteo API."""

import logging

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from wind_up.caching import with_parquet_cache
from wind_up.reanalysis_data import ReanalysisDataset

from .settings import get_cache_dir

logger = logging.getLogger(__name__)

HOT_LAT: float = 57.50
HOT_LON: float = -3.25
HOT_ERA5_START: str = "2000-01-01"
HOT_ERA5_END: str = "2026-05-01"
HOT_ERA5_FIELDS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "pressure_msl",
    "surface_pressure",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_10m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "weather_code",
]

_era5_cache_dir = get_cache_dir() / "era5_data"
_era5_cache_dir.mkdir(parents=True, exist_ok=True)


def _build_era5_df(response: object, fields: list[str]) -> pd.DataFrame:
    """Build a tidy hourly DataFrame from a single Open-Meteo response object."""
    hourly_data = response.Hourly()  # type: ignore[attr-defined]  # openmeteo_requests has no type stubs
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(
                start=pd.to_datetime(hourly_data.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly_data.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly_data.Interval()),
                inclusive="left",
            )
        }
        | {field: hourly_data.Variables(i).ValuesAsNumpy() for i, field in enumerate(fields)}
    ).set_index("timestamp")


@with_parquet_cache(_era5_cache_dir / f"ERA5_{HOT_LAT:.2f}_{HOT_LON:.2f}.parquet")
def get_hot_era5_hourly_df(
    lat: float = HOT_LAT,
    lon: float = HOT_LON,
    start_date: str = HOT_ERA5_START,
    end_date: str = HOT_ERA5_END,
    fields: list[str] = HOT_ERA5_FIELDS,
) -> pd.DataFrame:
    """Fetch hourly ERA5 data from Open-Meteo and return as a DataFrame.

    Warning: the cache path is fixed regardless of parameters. Calling with
    non-default arguments returns the cached default result if the cache file
    already exists. Delete the cache file to force a fresh fetch with new params.
    """
    openmeteo = openmeteo_requests.Client(
        session=retry(
            requests_cache.CachedSession(
                str(get_cache_dir() / "openmeteo_requests_cache"),
                expire_after=3600,
            ),
            retries=5,
            backoff_factor=0.2,
        )
    )
    responses = openmeteo.weather_api(
        url="https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": fields,
            "models": "era5",
            "wind_speed_unit": "ms",
        },
    )
    return _build_era5_df(responses[0], fields)


def get_hot_reanalysis_datasets() -> list[ReanalysisDataset]:
    """Return a list of ReanalysisDataset objects for the HOT site."""
    return [
        ReanalysisDataset(
            id=f"ERA5_{HOT_LAT:.2f}_{HOT_LON:.2f}",
            data=get_hot_era5_hourly_df(),
        )
    ]
