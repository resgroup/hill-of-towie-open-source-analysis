"""input data loading functions."""

import logging
from pathlib import Path

import pandas as pd
from wind_up.caching import with_parquet_cache

from .paths import DATA_DIR
from .scada_helpers import load_hot_10min_data, scada_df_to_wind_up_df
from .settings import get_cache_dir

logger = logging.getLogger(__name__)

DATASET_START = pd.Timestamp("2016-01-01", tz="UTC")  # open source dataset start
DATASET_V1_END_EXCL = pd.Timestamp("2024-09-01", tz="UTC")  # open source dataset v1.0 end
DATASET_V2_END_EXCL = pd.Timestamp("2026-05-01", tz="UTC")  # open source dataset v2.0 end

parquet_cache_dir = get_cache_dir() / "unpack_scada_data"
parquet_cache_dir.mkdir(parents=True, exist_ok=True)
logger.debug("Using parquet cache directory: %s", parquet_cache_dir)


@with_parquet_cache(parquet_cache_dir / "v1_scada_df.parquet")
def unpack_local_scada_data_v1(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Unpack Hill of Towie open source SCADA data."""
    scada_df = load_hot_10min_data(
        data_dir=data_dir,
        wtg_numbers=list(range(1, 22)),
        start_dt=DATASET_START,
        end_dt_excl=DATASET_V1_END_EXCL,
        rename_cols_using_aliases=True,
    )
    shutdown_duration_df = pd.read_csv(data_dir / "Hill_of_Towie_ShutdownDuration.zip", index_col=0, parse_dates=[0])
    return scada_df_to_wind_up_df(scada_df, shutdown_duration_df=shutdown_duration_df)


@with_parquet_cache(parquet_cache_dir / "v2_scada_df.parquet")
def unpack_local_scada_data_v2(
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Unpack Hill of Towie open source SCADA data."""
    scada_df = load_hot_10min_data(
        data_dir=data_dir,
        wtg_numbers=list(range(1, 22)),
        start_dt=DATASET_START,
        end_dt_excl=DATASET_V2_END_EXCL,
        rename_cols_using_aliases=True,
    )
    return scada_df_to_wind_up_df(scada_df)


@with_parquet_cache(parquet_cache_dir / "metadata_df.parquet")
def unpack_local_meta_data(data_dir: Path = DATA_DIR, scada_index_name: str = "TimeStamp_StartFormat") -> pd.DataFrame:
    """Unpack Hill of Towie open source turbine metadata."""
    timestamp_format = (
        "Start"
        if scada_index_name == "TimeStamp_StartFormat"
        else "End"
        if scada_index_name == "dtTimeStamp"
        else "Unknown"
    )
    return (
        pd.read_csv(data_dir / "Hill_of_Towie_turbine_metadata.csv")
        .loc[:, ["Turbine Name", "Latitude", "Longitude"]]
        .rename(columns={"Turbine Name": "Name"})
        .assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat=timestamp_format)
    )
