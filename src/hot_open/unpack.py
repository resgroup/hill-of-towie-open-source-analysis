"""input data loading functions."""

import logging
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from .fastlog_helpers import create_consistent_hash
from .parquet_io import write_parquet_atomic
from .scada_helpers import load_hot_10min_data, scada_df_to_wind_up_df
from .settings import get_cache_dir, get_data_dir
from .sourcing_data import ensure_hot_data_files

logger = logging.getLogger(__name__)

DATASET_START = pd.Timestamp("2016-01-01", tz="UTC")  # open source dataset start
DATASET_V1_END_EXCL = pd.Timestamp("2024-09-01", tz="UTC")  # open source dataset v1.0 end
DATASET_V2_END_EXCL = pd.Timestamp("2026-05-01", tz="UTC")  # open source dataset v2.0 end


def _cached(name: str, build: Callable[[], pd.DataFrame], **key: object) -> pd.DataFrame:
    cache_path = get_cache_dir() / "unpack_scada_data" / f"{name}_{create_consistent_hash(**key)}.parquet"
    if not cache_path.is_file():
        write_parquet_atomic(build(), cache_path)
    logger.info("Reading: %s", cache_path)
    return pd.read_parquet(cache_path)


def unpack_local_scada_data_v1(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Unpack Hill of Towie open source SCADA data."""
    data_dir = data_dir if data_dir is not None else get_data_dir()

    def build() -> pd.DataFrame:
        ensure_hot_data_files(["Hill_of_Towie_ShutdownDuration.zip"], data_dir=data_dir)
        scada_df = load_hot_10min_data(
            data_dir=data_dir,
            wtg_numbers=list(range(1, 22)),
            start_dt=DATASET_START,
            end_dt_excl=DATASET_V1_END_EXCL,
            rename_cols_using_aliases=True,
        )
        shutdown_path = data_dir / "Hill_of_Towie_ShutdownDuration.zip"
        logger.info("Reading: %s", shutdown_path)
        shutdown_duration_df = pd.read_csv(shutdown_path, index_col=0, parse_dates=[0])
        return scada_df_to_wind_up_df(scada_df, shutdown_duration_df=shutdown_duration_df)

    return _cached("v1_scada_df", build, data_dir=data_dir, start_dt=DATASET_START, end_dt_excl=DATASET_V1_END_EXCL)


def unpack_local_scada_data_v2(
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Unpack Hill of Towie open source SCADA data."""
    data_dir = data_dir if data_dir is not None else get_data_dir()

    def build() -> pd.DataFrame:
        scada_df = load_hot_10min_data(
            data_dir=data_dir,
            wtg_numbers=list(range(1, 22)),
            start_dt=DATASET_START,
            end_dt_excl=DATASET_V2_END_EXCL,
            rename_cols_using_aliases=True,
        )
        return scada_df_to_wind_up_df(scada_df)

    return _cached("v2_scada_df", build, data_dir=data_dir, start_dt=DATASET_START, end_dt_excl=DATASET_V2_END_EXCL)


def unpack_local_meta_data(
    data_dir: Path | None = None, scada_index_name: str = "TimeStamp_StartFormat"
) -> pd.DataFrame:
    """Unpack Hill of Towie open source turbine metadata."""
    data_dir = data_dir if data_dir is not None else get_data_dir()

    def build() -> pd.DataFrame:
        ensure_hot_data_files(["Hill_of_Towie_turbine_metadata.csv"], data_dir=data_dir)
        timestamp_format = (
            "Start"
            if scada_index_name == "TimeStamp_StartFormat"
            else "End"
            if scada_index_name == "dtTimeStamp"
            else "Unknown"
        )
        metadata_path = data_dir / "Hill_of_Towie_turbine_metadata.csv"
        logger.info("Reading: %s", metadata_path)
        return (
            pd.read_csv(metadata_path)
            .loc[:, ["Turbine Name", "Latitude", "Longitude"]]
            .rename(columns={"Turbine Name": "Name"})
            .assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat=timestamp_format)
        )

    return _cached("metadata_df", build, data_dir=data_dir, scada_index_name=scada_index_name)
