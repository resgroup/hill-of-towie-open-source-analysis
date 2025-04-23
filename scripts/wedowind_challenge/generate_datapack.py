# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "polars",
# ]
# ///

"""Generating the data-pack for the WeDoWind challenge.

It includes data from Turbines 1 (target), 2, 3, 4, 5, 7.
- Training set: 2016-2019.
- Test set: 2020.
"""

from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import polars as pl

from hot_open import setup_logger
from hot_open.sourcing_data import download_zenodo_data, get_analysis_directory

INDEX_FIELDS = {"TimeStamp": pl.Datetime(), "StationId": pl.Int32()}
STATION_TO_TURBINE_MAP = {x + 2304509: x for x in range(1, 21 + 1)}
FIELDS_DEFINITIONS = [
    {"field_name": "wtc_ActPower_mean", "table_name": "tblSCTurGrid", "dtype": pl.Float64()},
    {"field_name": "wtc_ActPower_stddev", "table_name": "tblSCTurGrid", "dtype": pl.Float64()},
    {"field_name": "wtc_AcWindSp_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_AcWindSp_stddev", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_NacelPos_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_NacelPos_min", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_NacelPos_max", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_GenRpm_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosA_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosB_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_PitcPosC_mean", "table_name": "tblSCTurbine", "dtype": pl.Float64()},
    {"field_name": "wtc_AmbieTmp_mean", "table_name": "tblSCTurTemp", "dtype": pl.Float64()},
    {"field_name": "wtc_ScReToOp_timeon", "table_name": "tblSCTurFlag", "dtype": pl.Float64()},
]
target_turbine = 1
target_field = "wtc_ActPower_mean"
ref_turbines = [2, 3, 4, 5, 7]


def _make_station_to_turbine_map(selected_turbines: set[int]) -> dict[int, int]:
    return {x + 2304509: x for x in selected_turbines}


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
            .select(_field_types)
            .filter(pl.col("StationId").is_in(station_to_turbine_map.keys()))
            for fname in fnames
        )
        frames_to_combine.append(_df)

    if not frames_to_combine:
        _msg = "No data found!"
        raise ValueError(_msg)

    combined_df = frames_to_combine[0]
    for _df in frames_to_combine[1:]:
        combined_df = combined_df.join(_df, how="full", on=index_cols, coalesce=True)

    return combined_df.with_columns(turbine_id=pl.col("StationId").replace(station_to_turbine_map)).drop("StationId")


if __name__ == "__main__":
    analysis_dir = get_analysis_directory(analysis_name="hill-of-towie-open-source-analysis")
    output_dir = analysis_dir / "wedowind_competition_input_data"
    output_dir.mkdir(exist_ok=True)

    # Download and Cache Source Data
    data_dir = analysis_dir / "zenodo_data"
    setup_logger(data_dir / f"{Path(__file__).stem}.log")
    download_zenodo_data(
        record_id="14870023",
        output_dir=data_dir,
        filenames=[
            *[f"{x}.zip" for x in range(2016, 2020)],
            "Hill_of_Towie_ShutdownDuration.zip",
            "Hill_of_Towie_turbine_metadata.csv",
        ],
    )

    # Train Dataset
    (
        pl.concat(
            _extract_data_from_year_zipfile(
                year_data_zip_fpath=data_dir / f"{year}.zip",
                index_fields=INDEX_FIELDS,
                field_definitions=FIELDS_DEFINITIONS,
                station_to_turbine_map=_make_station_to_turbine_map({target_turbine, *ref_turbines}),
            )
            for year in (2016, 2017, 2018, 2019)
        )
        .group_by(["turbine_id", "TimeStamp"])
        .mean()  # ensure unique timestamp turbine combination
        .sink_parquet(output_dir / "train_dataset.parquet")
    )

    # Evaluation Dataset
    (
        _extract_data_from_year_zipfile(
            year_data_zip_fpath=data_dir / "2020.zip",
            index_fields=INDEX_FIELDS,
            field_definitions=FIELDS_DEFINITIONS,
            station_to_turbine_map=_make_station_to_turbine_map(set(ref_turbines)),
        )
        .group_by(["turbine_id", "TimeStamp"])
        .mean()  # ensure unique timestamp turbine combination
        .sink_parquet(output_dir / "eval_dataset.parquet")
    )
