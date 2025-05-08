"""input data loading functions."""

from pathlib import Path

import pandas as pd
from wind_up.caching import with_parquet_cache

from .helpers import load_hot_10min_data, scada_df_to_wind_up_df
from .paths import DATA_DIR, GLOBAL_CACHE_DIR

DATASET_START = pd.Timestamp("2016-01-01", tz="UTC")  # open source dataset start
DATASET_END_EXCL = pd.Timestamp("2024-09-01", tz="UTC")  # open source dataset end


@with_parquet_cache(GLOBAL_CACHE_DIR / "scada_df.parquet")
def unpack_local_scada_data(
    data_dir: Path = DATA_DIR,
    *,
    start_dt: pd.Timestamp = DATASET_START,
    end_dt_excl: pd.Timestamp = DATASET_END_EXCL,
) -> pd.DataFrame:
    """Unpack Hill of Towie open source SCADA data."""
    scada_df = load_hot_10min_data(
        data_dir=data_dir,
        wtg_numbers=list(range(1, 22)),
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
    )
    shutdown_duration_df = pd.read_csv(data_dir / "Hill_of_Towie_ShutdownDuration.zip", index_col=0, parse_dates=[0])
    return scada_df_to_wind_up_df(scada_df, shutdown_duration_df=shutdown_duration_df)


@with_parquet_cache(GLOBAL_CACHE_DIR / "metadata_df.parquet")
def unpack_local_meta_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Unpack Hill of Towie open source turbine metadata."""
    return (
        pd.read_csv(data_dir / "Hill_of_Towie_turbine_metadata.csv")
        .loc[:, ["Turbine Name", "Latitude", "Longitude"]]
        .rename(columns={"Turbine Name": "Name"})
        .assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")
    )
