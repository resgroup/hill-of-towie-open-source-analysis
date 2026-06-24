"""ERA5 reanalysis data fetching via Open-Meteo API."""

import hashlib
import json
import logging
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from wind_up.reanalysis_data import ReanalysisDataset

from .settings import get_cache_dir

logger = logging.getLogger(__name__)

HOT_LAT: float = 57.50
HOT_LON: float = -3.25
HOT_ERA5_START: str = "2000-01-01"
HOT_ERA5_END: str = "2026-05-01"
ERA5_DEFAULT_FIELDS: list[str] = [
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
HOT_ERA5_FIELDS: list[str] = ERA5_DEFAULT_FIELDS

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


def _era5_cache_path(*, lat: float, lon: float, start_date: str, end_date: str, fields: list[str]) -> Path:
    """Build a deterministic parquet cache path from the request args."""
    args_blob = json.dumps(
        {"lat": lat, "lon": lon, "start_date": start_date, "end_date": end_date, "fields": list(fields)},
        sort_keys=True,
    )
    args_hash = hashlib.sha256(args_blob.encode("utf-8")).hexdigest()[:16]
    return _era5_cache_dir / f"ERA5_{lat:.2f}_{lon:.2f}_{start_date}_{end_date}_{args_hash}.parquet"


def get_era5_hourly_df(
    *,
    lat: float,
    lon: float,
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    fields: list[str] = ERA5_DEFAULT_FIELDS,
) -> pd.DataFrame:
    """Fetch hourly ERA5 data from Open-Meteo for any location and return as a DataFrame.

    ``end_date`` defaults to today (UTC) when ``None``. Each unique combination of arguments
    is cached to its own parquet file keyed by a hash of the arguments. Delete the cache file
    to force a refetch.
    """
    if end_date is None:
        end_date = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
    cache_path = _era5_cache_path(lat=lat, lon=lon, start_date=start_date, end_date=end_date, fields=fields)
    if cache_path.exists():
        logger.info("Reading: %s", cache_path)
        return pd.read_parquet(cache_path)

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
    df = _build_era5_df(responses[0], fields)
    logger.info("Writing: %s", cache_path)
    df.to_parquet(cache_path)
    return df


def get_hot_era5_hourly_df(
    *,
    start_date: str = HOT_ERA5_START,
    end_date: str = HOT_ERA5_END,
    fields: list[str] = HOT_ERA5_FIELDS,
) -> pd.DataFrame:
    """Fetch hourly ERA5 data for the HOT site (convenience wrapper over :func:`get_era5_hourly_df`)."""
    return get_era5_hourly_df(lat=HOT_LAT, lon=HOT_LON, start_date=start_date, end_date=end_date, fields=fields)


def get_hot_reanalysis_datasets() -> list[ReanalysisDataset]:
    """Return a list of ReanalysisDataset objects for the HOT site."""
    return [
        ReanalysisDataset(
            id=f"ERA5_{HOT_LAT:.2f}_{HOT_LON:.2f}",
            data=get_hot_era5_hourly_df(),
        )
    ]
