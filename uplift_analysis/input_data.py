"""input data loading functions."""

from pathlib import Path

import pandas as pd
from helpers import load_hot_10min_data, scada_df_to_wind_up_df
from wind_up.caching import with_parquet_cache

OUT_DIR = Path.home() / "temp" / "hill-of-towie-open-source-analysis" / Path(__file__).stem
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = Path(__file__).parent / "wind_up_config"


@with_parquet_cache(CACHE_DIR / "scada_df.parquet")
def unpack_local_scada_data(data_dir: Path) -> pd.DataFrame:
    """Unpack Hill of Towie open source SCADA data."""
    scada_df = load_hot_10min_data(
        data_dir=data_dir,
        wtg_numbers=list(range(1, 22)),
        start_dt=pd.Timestamp("2016-01-01", tz="UTC"),  # open source dataset start
        end_dt_excl=pd.Timestamp("2024-09-01", tz="UTC"),  # open source dataset end
    )
    shutdown_duration_df = pd.read_csv(data_dir / "Hill_of_Towie_ShutdownDuration.zip", index_col=0, parse_dates=[0])
    return scada_df_to_wind_up_df(scada_df, shutdown_duration_df=shutdown_duration_df)


@with_parquet_cache(CACHE_DIR / "metadata_df.parquet")
def unpack_local_meta_data(data_dir: Path) -> pd.DataFrame:
    """Unpack Hill of Towie open source turbine metadata."""
    return (
        pd.read_csv(data_dir / "Hill_of_Towie_turbine_metadata.csv")
        .loc[:, ["Turbine Name", "Latitude", "Longitude"]]
        .rename(columns={"Turbine Name": "Name"})
        .assign(TimeZone="UTC", TimeSpanMinutes=10, TimeFormat="Start")
    )
