"""Analyse the uplift of the T13 AeroUp upgrade."""

import getpass
from pathlib import Path

import pandas as pd
from wind_up.constants import DataColumns

from helpers import load_hot_10min_data

username = getpass.getuser()

scada_df = load_hot_10min_data(
    data_dir=Path(r"C:\Users") / username / "RES Group/Digital Solutions - HardTech - Open source dataset",
    wtg_numbers=list(range(1, 22)),
    start_dt=pd.Timestamp("2024-01-01", tz="UTC"),
    end_dt_excl=pd.Timestamp("2024-08-31", tz="UTC"),
)
wind_up_df = (
    scada_df.stack(level=0, future_stack=True).reset_index(level=1).rename(columns={"StationId": "TurbineName"})  # noqa:PD013
)

if DataColumns.pitch_angle_mean not in wind_up_df.columns:
    wind_up_df[DataColumns.pitch_angle_mean] = wind_up_df[["pitch_angle_a", "pitch_angle_b", "pitch_angle_c"]].mean(
        axis=1
    )

if DataColumns.shutdown_duration not in wind_up_df.columns:
    wind_up_df[DataColumns.shutdown_duration] = (
        600 - wind_up_df[["Time ready to operate in period", "Time running in period"]].max(axis=1)
    ).fillna(0)
