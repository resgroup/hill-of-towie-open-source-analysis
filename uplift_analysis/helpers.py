"""Helpful things."""

import logging
import math
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple
from zipfile import ZipFile

import pandas as pd
import requests
from wind_up.constants import DataColumns

logger = logging.getLogger(__name__)

BYTES_IN_10MB = 10 * 1024 * 1024


def download_zenodo_data(
    record_id: str, output_dir: Path, filenames: Collection[str] | None = None, *, cache_overwrite: bool = False
) -> None:
    """Donwload files from a Zenodo record."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=10)
        r.raise_for_status()

    # Catch all exceptions
    except Exception as e:
        msg = f"Failed to download Zenodo record {record_id}. Error: {e}"
        logger.info(msg)
        raise

    files_to_download = r.json()["files"]
    if filenames is not None:
        files_to_download = [i for i in files_to_download if i["key"] in set(filenames)]
        if len(files_to_download) != len(filenames):
            msg = (
                f"Could not find all files in the Zenodo record. "
                f"Missing files: {set(filenames) - {i['key'] for i in files_to_download}}"
            )
            raise ValueError(msg)

    filepaths = []
    for file_to_download in files_to_download:
        dst_fpath = output_dir / file_to_download["key"]
        if not dst_fpath.exists() or cache_overwrite:
            msg = f"Beginning file download from Zenodo: {file_to_download['key']}..."
            logger.info(msg)
            filesize = file_to_download["size"] / BYTES_IN_10MB
            result = requests.get(file_to_download["links"]["self"], stream=True, timeout=10)
            chunk_number = 0
            with Path.open(dst_fpath, "wb") as f:
                for chunk in result.iter_content(chunk_size=BYTES_IN_10MB):
                    chunk_number = chunk_number + 1
                    msg = f"{10 * chunk_number} out of {10 * math.ceil(filesize)} MB downloaded"
                    logger.info(msg)
                    f.write(chunk)
        else:
            msg = f"File {dst_fpath} already exists. Skipping download."
            logger.info(msg)
        filepaths.append(dst_fpath)


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
    WPSBackupFileField(
        alias="Time ready to operate in period", field_name="wtc_ScReToOp_timeon", table_name="tblSCTurFlag"
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
            [{x: f"T{x - 2304509:02d}" for x in cols.get_level_values(0).unique()}[x] for x in cols.levels[0]],  # type:ignore[attr-defined]
            level=0,
        )
    return (
        combined_df[(combined_df.index >= start_dt) & (combined_df.index < end_dt_excl)]
        .resample(pd.Timedelta(seconds=timebase_s))
        .first()
    )


def calc_shutdown_duration(wind_up_df: pd.DataFrame) -> pd.DataFrame:
    """Caclulate ShutdownDuration column and return the wind up dataframe with the new column."""
    wind_up_df = wind_up_df.copy()
    timebase_s = 600
    wind_up_df[DataColumns.shutdown_duration] = timebase_s - wind_up_df["Time ready to operate in period"].fillna(
        timebase_s
    )
    # stuck data is believed to be an indicator of downtime
    diffdf = (
        wind_up_df[
            [
                "TurbineName",
                DataColumns.active_power_mean,
                DataColumns.active_power_sd,
                DataColumns.wind_speed_mean,
                DataColumns.wind_speed_sd,
                DataColumns.gen_rpm_mean,
                DataColumns.pitch_angle_mean,
                DataColumns.yaw_angle_mean,
            ]
        ]
        .groupby("TurbineName", observed=False)
        .ffill()
        .fillna(0)
        .diff()
    )
    stuck_data = (diffdf == 0).all(axis=1)
    very_low_wind_threshold = 1.5
    very_low_wind = wind_up_df[DataColumns.wind_speed_mean] < very_low_wind_threshold
    stuck_filter = stuck_data & (~very_low_wind)
    wind_up_df.loc[stuck_filter, DataColumns.shutdown_duration] = timebase_s
    return wind_up_df


def scada_df_to_wind_up_df(scada_df: pd.DataFrame, *, shutdown_duration_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Convert wide format scada_df to narrow format wind_up_df.

    scada_df has two column levels and wind_up_df has one column level and TurbineName column.
    shutdown_duration_df is optional, if provided merge in ShutdownDuration column from this dataframe.
    """
    wind_up_df = (
        scada_df.stack(level=0, future_stack=True).reset_index(level=1).rename(columns={"StationId": "TurbineName"})  # noqa:PD013
    )

    if DataColumns.pitch_angle_mean not in wind_up_df.columns:
        wind_up_df[DataColumns.pitch_angle_mean] = wind_up_df[["pitch_angle_a", "pitch_angle_b", "pitch_angle_c"]].mean(
            axis=1
        )
    if shutdown_duration_df is not None:
        wind_up_df = wind_up_df.merge(shutdown_duration_df, how="left", on=["TimeStamp_StartFormat", "TurbineName"])
    else:
        wind_up_df = calc_shutdown_duration(wind_up_df)
    return wind_up_df


def setup_logger(log_fpath: Path, level: int = logging.INFO) -> None:
    """Initialize the logger with a file handler and a console handler."""
    log_formatter_file = logging.Formatter("%(asctime)s [%(levelname)-8s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_handler = logging.FileHandler(log_fpath, mode="w")
    file_handler.setFormatter(log_formatter_file)
    root_logger.addHandler(file_handler)

    log_formatter_console = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter_console)
    root_logger.addHandler(console_handler)
