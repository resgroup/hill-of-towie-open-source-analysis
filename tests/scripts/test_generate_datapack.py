import pandas as pd
import polars as pl

from hot_open.helpers import WPSBackupFileField, load_hot_10min_data
from scripts.wedowind_challenge.s02_generate_datapack import INDEX_FIELDS, _extract_data_from_year_zipfile
from tests.conftest import TEST_DATA_DIR


def test_extract_data_from_year_zipfile_similar_to_load_hot_10min_data() -> None:
    """Test that _extract_data_from_year_zipfile output is similar to load_hot_10min_data.

    There are some differences in the output, but they are not significant for the purpose
    of this test and are corrected, see `actual_lazy` to `actual` transformation
    """
    wtg_numbers = [1, 2]
    fields = ["wtc_ScReToOp_timeon", "wtc_ScInOper_timeon"]

    actual_lazy = _extract_data_from_year_zipfile(
        year_data_zip_fpath=TEST_DATA_DIR / "2024.zip",
        index_fields=INDEX_FIELDS,
        field_definitions=[{"field_name": i, "table_name": "tblSCTurFlag", "dtype": pl.Float64()} for i in fields],
        station_to_turbine_map={x + 2304509: x for x in wtg_numbers},
    )

    def _reintroduce_station_ids(d: pl.LazyFrame) -> pl.LazyFrame:
        return d.with_columns(StationId="T0" + pl.col("turbine_id").cast(pl.String)).drop("turbine_id")

    def _drop_timestamps_not_in_the_month(d: pl.LazyFrame) -> pl.LazyFrame:
        # load_hot_10min_data drops timestamps not in this month when converted to StartFormat
        return d.filter(pl.col("TimeStamp_StartFormat") >= pd.Timestamp("2024-07-01", tz="UTC"))

    def _reshape(d: pd.DataFrame) -> pd.DataFrame:
        return (
            d.pivot_table(
                index=["TimeStamp_StartFormat"],
                columns=["StationId"],
                dropna=False,
            ).swaplevel(axis=1)  # type: ignore[call-arg,return-value]
        )

    def _cast_timeindex(d: pd.DataFrame) -> pd.DataFrame:
        """Force index to be in nano-seconds."""
        d.index = d.index.as_unit("ns")  # type: ignore[attr-defined]
        return d

    actual = (
        actual_lazy.pipe(_reintroduce_station_ids)
        .pipe(_drop_timestamps_not_in_the_month)
        .collect()
        .to_pandas()
        .pipe(_reshape)
        .pipe(_cast_timeindex)
    )

    expected = load_hot_10min_data(
        data_dir=TEST_DATA_DIR,
        wtg_numbers=wtg_numbers,
        start_dt=pd.Timestamp("2024-06-01", tz="UTC"),
        end_dt_excl=pd.Timestamp("2024-09-02", tz="UTC"),
        custom_fields=[WPSBackupFileField(alias=i, field_name=i, table_name="tblSCTurFlag") for i in fields],
    )
    pd.testing.assert_frame_equal(actual, expected, check_like=True)
