"""Helpful things."""

from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile

import pandas as pd
from wind_up.constants import DataColumns


class WPSBackupFileField(NamedTuple):
    """Class for Hill of Towie field and table mappings."""

    alias: str
    field_name: str
    table_name: str


hill_of_towie_fields = [
    WPSBackupFileField(alias=DataColumns.active_power_mean, field_name="wtc_ActPower_mean", table_name="tblSCTurGrid"),
    WPSBackupFileField(alias=DataColumns.active_power_sd, field_name="wtc_ActPower_stddev", table_name="tblSCTurGrid"),
    WPSBackupFileField(alias=DataColumns.wind_speed_mean, field_name="wtc_AcWindSp_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.wind_speed_sd, field_name="wtc_AcWindSp_stddev", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.yaw_angle_mean, field_name="wtc_NacelPos_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.yaw_angle_min, field_name="wtc_NacelPos_min", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.yaw_angle_max, field_name="wtc_NacelPos_max", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.gen_rpm_mean, field_name="wtc_GenRpm_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias="pitch_angle_a", field_name="wtc_PitcPosA_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias="pitch_angle_b", field_name="wtc_PitcPosB_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias="pitch_angle_c", field_name="wtc_PitcPosC_mean", table_name="tblSCTurbine"),
    WPSBackupFileField(alias=DataColumns.ambient_temp, field_name="wtc_AmbieTmp_mean", table_name="tblSCTurTemp"),
    WPSBackupFileField(alias="Time running in period", field_name="wtc_ScInOper_timeon", table_name="tblSCTurFlag"),
    WPSBackupFileField(
        alias="Time ready to operate in period", field_name="wtc_ScReToOp_timeon", table_name="tblSCTurFlag"
    ),
    WPSBackupFileField(
        alias="Time turbine error active in period", field_name="wtc_ScTurSto_timeon", table_name="tblSCTurFlag"
    ),
]


def load_hot_10min_data(  # noqa:PLR0913 C901
    *,
    data_dir: Path,
    wtg_numbers: list[int],
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    use_turbine_names: bool = True,  # if False serial numbers are used to identify turbines
    rename_cols_using_aliases: bool = True,
) -> pd.DataFrame:
    """Return a SCADA 10-min dataframe of for Hill of Towie."""
    if str(start_dt.tz) != "UTC" or str(end_dt_excl.tz) != "UTC":
        msg = "start_dt and end_dt_excl must be in UTC"
        raise ValueError(msg)

    timebase_s = 600

    serial_numbers = [x + 2304509 for x in wtg_numbers]

    first_year_to_load = start_dt.year
    last_year_to_load = (end_dt_excl - pd.Timedelta(seconds=timebase_s)).year
    years_to_load = list(range(first_year_to_load, last_year_to_load + 1))
    tables_to_load = {x.table_name for x in hill_of_towie_fields}
    result_dfs = []
    for _year in years_to_load:
        zip_path = data_dir / f"{_year}.zip"
        with ZipFile(zip_path) as zip_file:
            year_dfs = []
            for _table in tables_to_load:
                table_dfs = []
                for _month in range(1, 13):
                    if pd.Timestamp(year=_year, month=_month, day=1, tz="UTC") < (
                        start_dt - pd.DateOffset(months=1, days=1)
                    ) or pd.Timestamp(year=_year, month=_month, day=1, tz="UTC") > (
                        end_dt_excl + pd.DateOffset(months=1, days=1)
                    ):
                        continue
                    if (fname := f"{_table}_{_year}_{_month:02d}.csv") not in zip_file.namelist():
                        continue
                    _df = pd.read_csv(zip_file.open(fname), index_col=0, parse_dates=True)[
                        ["StationId", *[x.field_name for x in hill_of_towie_fields if x.table_name == _table]]
                    ]
                    if rename_cols_using_aliases:
                        _df = _df.rename(
                            columns={x.field_name: x.alias for x in hill_of_towie_fields if x.table_name == _table}
                        )
                    if _df.index.name != "TimeStamp":
                        msg = f"unexpected index name, {_df.index.name =}"
                        raise ValueError(msg)
                    if not isinstance(_df.index, pd.DatetimeIndex):
                        # try to convert it again
                        _df.index = pd.to_datetime(_df.index, format="ISO8601")
                        if not isinstance(_df.index, pd.DatetimeIndex):
                            msg = f"unexpected index type, {_df.index.name =} {type(_df.index)=}"
                            raise TypeError(msg)
                    # convert to Start Format UTC
                    _df.index = _df.index.tz_localize("UTC")  # type:ignore[attr-defined]
                    _df.index = _df.index - pd.Timedelta(minutes=10)
                    _df.index.name = "TimeStamp_StartFormat"
                    # drop any timestamps not in this month; apparently the files overlap by 10 minutes
                    _df = _df[(_df.index.year == _year) & (_df.index.month == _month)]  # type:ignore[attr-defined,assignment]
                    # drop any unwanted turbines
                    _df = _df[_df["StationId"].isin(serial_numbers)]
                    pivoted_df = _df.pivot_table(
                        index=_df.index.name,
                        columns="StationId",
                        values=[x for x in _df.columns if x != "StationId"],
                    ).swaplevel(axis=1)
                    table_dfs.append(pivoted_df)
                table_df = pd.concat(table_dfs, verify_integrity=True, sort=True)
                year_dfs.append(table_df)
            year_df = pd.concat(year_dfs, axis=1)
            result_dfs.append(year_df)
    combined_df = pd.concat(result_dfs, verify_integrity=True, sort=True)
    if use_turbine_names:
        cols = combined_df.columns
        combined_df.columns = cols.set_levels(  # type:ignore[attr-defined]
            [{x: f"T{x-2304509:02d}" for x in cols.get_level_values(0).unique()}[x] for x in cols.levels[0]],  # type:ignore[attr-defined]
            level=0,
        )
    return combined_df[(combined_df.index >= start_dt) & (combined_df.index < end_dt_excl)]
