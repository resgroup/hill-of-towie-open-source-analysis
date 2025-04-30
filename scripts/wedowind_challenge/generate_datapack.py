# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "hot-open",
#     "polars",
# ]
#
# [tool.uv.sources]
# hot-open = { path = "../../" }
# ///

"""Generating the data-pack for the WeDoWind challenge.

It includes data from Turbines 1 (target), 2, 3, 4, 5, 7.
- Training set: 2016-2019.
- Test set: 2020.
"""

import datetime as dt
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import polars as pl
from dotenv import load_dotenv

from hot_open import setup_logger
from hot_open.paths import ANALYSES_DIR, DATA_DIR
from hot_open.sourcing_data import download_zenodo_data

load_dotenv()

INDEX_FIELDS = {"TimeStamp": pl.Datetime(time_zone="UTC"), "StationId": pl.Int32()}
STATION_TO_TURBINE_MAP = {x + 2304509: x for x in range(1, 21 + 1)}
FIELDS_DEFINITIONS = [
    {"field_name": "wtc_ActPower_mean", "table_name": "tblSCTurGrid", "dtype": pl.Float64()},
    {"field_name": "wtc_ActPower_min", "table_name": "tblSCTurGrid", "dtype": pl.Float64()},
    {"field_name": "wtc_ActPower_max", "table_name": "tblSCTurGrid", "dtype": pl.Float64()},
    {"field_name": "wtc_ActPower_stddev", "table_name": "tblSCTurGrid", "dtype": pl.Float64()},
    {"field_name": "wtc_AcWindSp_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_AcWindSp_min", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_AcWindSp_max", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_AcWindSp_stddev", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_ScYawPos_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_ScYawPos_min", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_ScYawPos_max", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_ScYawPos_stddev", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_NacelPos_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_NacelPos_min", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_NacelPos_max", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_GenRpm_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_GenRpm_min", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_GenRpm_max", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_GenRpm_stddev", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosA_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosA_min", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosA_max", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosA_stddev", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosB_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosC_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PowerRef_endvalue", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_AmbieTmp_mean", "table_name": "tblSCTurTemp", "dtype": pl.Float64()},
    {"field_name": "wtc_ScReToOp_timeon", "table_name": "tblSCTurFlag", "dtype": pl.Float64()},
]


def _make_station_to_turbine_map(selected_turbines: set[int]) -> dict[int, int]:
    return {x + 2304509: x for x in selected_turbines}


def _resample_10min(df: pl.LazyFrame) -> pl.LazyFrame:
    _start = df.select("TimeStamp_StartFormat").min().collect().item()
    _end = df.select("TimeStamp_StartFormat").max().collect().item()
    _ts_spine = pl.datetime_range(start=_start, end=_end, interval="10m", closed="both", eager=True).to_frame(
        "TimeStamp_StartFormat"
    )
    _turbines = df.select("turbine_id").unique().collect()
    _spine = _ts_spine.join(_turbines, how="cross").lazy()
    return (
        df.group_by(["turbine_id", "TimeStamp_StartFormat"])
        .mean()
        .join(_spine, on=["TimeStamp_StartFormat", "turbine_id"], how="right")
    )


def _extract_data_from_year_zipfile(
    year_data_zip_fpath: Path,
    index_fields: dict[str, pl.DataType],
    field_definitions: list[dict],
    station_to_turbine_map: dict[int, int],
) -> pl.LazyFrame:
    zf = ZipFile(year_data_zip_fpath)

    table_fields_dtype_map: dict[str, dict[str, pl.DataType]] = defaultdict(lambda: index_fields.copy())
    for row in field_definitions:
        table_fields_dtype_map[row["table_name"]][row["field_name"]] = row["dtype"]

    files_to_combine = defaultdict(list)
    for fname in zf.namelist():
        table = fname.split("_", maxsplit=1)[0]
        if table not in table_fields_dtype_map:
            continue
        files_to_combine[table].append(fname)

    index_cols = tuple(index_fields.keys())
    frames_to_combine = []
    for table, fnames in files_to_combine.items():
        _field_types = table_fields_dtype_map[table]
        _df = pl.concat(
            # using `schema` raises error that when ignored nullify entire columns
            pl.scan_csv(zf.open(fname), schema_overrides=pl.Schema(_field_types))
            .filter(pl.col("StationId").is_in(station_to_turbine_map.keys()))
            .select(_field_types)
            for fname in fnames
        )
        frames_to_combine.append(_df)

    if not frames_to_combine:
        _msg = "No data found!"
        raise ValueError(_msg)

    combined_df = frames_to_combine[0]
    for _df in frames_to_combine[1:]:
        combined_df = combined_df.join(_df, how="full", on=index_cols, coalesce=True)

    return (
        combined_df.with_columns(
            turbine_id=pl.col("StationId").replace(station_to_turbine_map),
            TimeStamp_StartFormat=pl.col("TimeStamp") - pl.duration(minutes=10),
        )
        .drop("StationId", "TimeStamp")
        .pipe(_resample_10min)
    )


def _extract_shutdown_data(zip_fpath: Path, turbine_ids: list[int]) -> pl.LazyFrame:
    turbine_name_mapping = {f"T{x:02}": x for x in turbine_ids}
    return (
        pl.scan_csv(
            ZipFile(zip_fpath).open("ShutdownDuration.csv").read(),
            schema_overrides=pl.Schema(
                {"TimeStamp_StartFormat": pl.Datetime(time_zone="UTC"), "ShutdownDuration": pl.Int16()}
            ),
        )
        .filter(pl.col("TurbineName").is_in(turbine_name_mapping.keys()))
        .with_columns(turbine_id=pl.col("TurbineName").replace(turbine_name_mapping).cast(pl.Int32()))
        .drop("TurbineName")
    )


def _remove_duplicates(df: pl.LazyFrame) -> pl.LazyFrame:
    _primary_keys = ["turbine_id", "TimeStamp_StartFormat"]
    return df.group_by(_primary_keys).mean().sort(_primary_keys)


def _augment_with_shutdown_data(df: pl.LazyFrame, shutdown_df: pl.LazyFrame) -> pl.LazyFrame:
    return df.join(shutdown_df, how="left", on=["turbine_id", "TimeStamp_StartFormat"])


if __name__ == "__main__":
    target_turbine = 1
    ref_turbines = [2, 3, 4, 5, 7]
    dataset_start = dt.datetime(2016, 1, 1, tzinfo=dt.timezone.utc)
    submission_data_start = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)

    output_dir = ANALYSES_DIR / "wedowind_competition_input_data"
    output_dir.mkdir(exist_ok=True)

    # Download and Cache Source Data
    setup_logger(DATA_DIR / f"{Path(__file__).stem}.log")
    download_zenodo_data(
        record_id="14870023",
        filenames=[
            *[f"{x}.zip" for x in range(2016, 2020)],
            "Hill_of_Towie_ShutdownDuration.zip",
            "Hill_of_Towie_turbine_metadata.csv",
        ],
    )

    shutdown_df = _extract_shutdown_data(
        zip_fpath=DATA_DIR / "Hill_of_Towie_ShutdownDuration.zip", turbine_ids=[target_turbine, *ref_turbines]
    )

    full_dataset = (
        pl.concat(
            _extract_data_from_year_zipfile(
                year_data_zip_fpath=DATA_DIR / f"{year}.zip",
                index_fields=INDEX_FIELDS,
                field_definitions=FIELDS_DEFINITIONS,
                station_to_turbine_map=_make_station_to_turbine_map({target_turbine, *ref_turbines}),
            )
            for year in (2016, 2017, 2018, 2019, 2020)
        )
        .pipe(_remove_duplicates)
        .pipe(_augment_with_shutdown_data, shutdown_df=shutdown_df)
        .sort("TimeStamp_StartFormat", "turbine_id")
    )

    # Train Dataset
    (
        full_dataset.filter(
            pl.col("TimeStamp_StartFormat") >= dataset_start,
            pl.col("TimeStamp_StartFormat") < submission_data_start,
        ).sink_parquet(output_dir / "train_dataset.parquet")
    )

    # Submission Dataset
    (
        full_dataset.filter(
            pl.col("TimeStamp_StartFormat") >= submission_data_start,
            pl.col("turbine_id").is_in(ref_turbines),
        ).sink_parquet(output_dir / "submission_dataset.parquet")
    )
