from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from wind_up.constants import DataColumns

import hot_open.unpack
from tests.synthetic_scada import field_value, write_shutdown_duration_zip, write_year_zips

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType

# straddles a year boundary so two year zips are read and concatenated
WINDOW_START = pd.Timestamp("2023-12-31 23:40", tz="UTC")
WINDOW_END_EXCL = pd.Timestamp("2024-01-01 00:20", tz="UTC")
N_TIMESTAMPS = 4
N_TURBINES = 21


@pytest.fixture
def unpack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    """hot_open.unpack with its parquet caches under tmp_path and a short dataset window."""
    monkeypatch.setenv("HOT_OPEN_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(hot_open.unpack, "DATASET_START", WINDOW_START)
    monkeypatch.setattr(hot_open.unpack, "DATASET_V1_END_EXCL", WINDOW_END_EXCL)
    monkeypatch.setattr(hot_open.unpack, "DATASET_V2_END_EXCL", WINDOW_END_EXCL)
    return hot_open.unpack


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """Year zips covering the test window with an hour either side, so the loader must trim them."""
    path = tmp_path / "data"
    path.mkdir()
    write_year_zips(
        path,
        start_dt=WINDOW_START - pd.Timedelta(hours=1),
        end_dt_excl=WINDOW_END_EXCL + pd.Timedelta(hours=1),
    )
    return path


def _assert_wind_up_shape(df: pd.DataFrame) -> None:
    assert len(df) == N_TIMESTAMPS * N_TURBINES
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "TimeStamp_StartFormat"
    assert str(df.index.tz) == "UTC"
    assert df.index.min() == WINDOW_START
    assert df.index.max() == WINDOW_END_EXCL - pd.Timedelta(minutes=10)
    assert sorted(df["TurbineName"].unique()) == [f"T{i:02d}" for i in range(1, 22)]
    for col in (
        DataColumns.active_power_mean,
        DataColumns.wind_speed_mean,
        DataColumns.yaw_angle_mean,
        DataColumns.pitch_angle_mean,
        DataColumns.gen_rpm_mean,
        DataColumns.shutdown_duration,
    ):
        assert col in df.columns
        assert pd.api.types.is_numeric_dtype(df[col])


def _assert_values_match_source(df: pd.DataFrame) -> None:
    row = df[(df["TurbineName"] == "T05") & (df.index == WINDOW_START)].iloc[0]
    assert row[DataColumns.active_power_mean] == field_value(
        field_name="wtc_ActPower_mean", wtg_number=5, start_ts=WINDOW_START
    )
    assert row[DataColumns.yaw_angle_mean] == field_value(
        field_name="wtc_NacelPos_mean", wtg_number=5, start_ts=WINDOW_START
    )
    pitch = [
        field_value(field_name=f"wtc_PitcPos{b}_mean", wtg_number=5, start_ts=WINDOW_START) for b in ("A", "B", "C")
    ]
    assert row[DataColumns.pitch_angle_mean] == pytest.approx(np.mean(pitch))


class TestUnpackLocalScadaDataV1:
    def test_returns_narrow_wind_up_frame(self, unpack: ModuleType, data_dir: Path) -> None:
        write_shutdown_duration_zip(data_dir, start_dt=WINDOW_START, end_dt_excl=WINDOW_END_EXCL)
        df = unpack.unpack_local_scada_data_v1(data_dir=data_dir)
        _assert_wind_up_shape(df)
        _assert_values_match_source(df)

    def test_shutdown_duration_comes_from_the_published_file(self, unpack: ModuleType, data_dir: Path) -> None:
        """v1 must merge the published ShutdownDuration, not recompute it from the SCADA signals."""
        written = write_shutdown_duration_zip(data_dir, start_dt=WINDOW_START, end_dt_excl=WINDOW_END_EXCL)
        df = unpack.unpack_local_scada_data_v1(data_dir=data_dir)
        expected = written.assign(
            TimeStamp_StartFormat=pd.to_datetime(written["TimeStamp_StartFormat"], utc=True)
        ).set_index(["TimeStamp_StartFormat", "TurbineName"])["ShutdownDuration"]
        actual = df.set_index("TurbineName", append=True)[DataColumns.shutdown_duration]
        pd.testing.assert_series_equal(actual.sort_index(), expected.sort_index(), check_names=False, check_dtype=False)

    def test_second_call_is_served_from_the_parquet_cache(
        self, unpack: ModuleType, data_dir: Path, tmp_path: Path
    ) -> None:
        write_shutdown_duration_zip(data_dir, start_dt=WINDOW_START, end_dt_excl=WINDOW_END_EXCL)
        first = unpack.unpack_local_scada_data_v1(data_dir=data_dir)
        assert len(list((tmp_path / "cache" / "unpack_scada_data").glob("v1_scada_df_*.parquet"))) == 1
        for f in data_dir.glob("*.zip"):
            f.unlink()
        second = unpack.unpack_local_scada_data_v1(data_dir=data_dir)
        pd.testing.assert_frame_equal(first, second)


class TestUnpackLocalScadaDataV2:
    def test_returns_narrow_wind_up_frame(self, unpack: ModuleType, data_dir: Path) -> None:
        df = unpack.unpack_local_scada_data_v2(data_dir=data_dir)
        _assert_wind_up_shape(df)
        _assert_values_match_source(df)

    def test_shutdown_duration_is_derived_from_scada(self, unpack: ModuleType, data_dir: Path) -> None:
        """Without a published ShutdownDuration, v2 derives it; varying signals at full availability give 0."""
        df = unpack.unpack_local_scada_data_v2(data_dir=data_dir)
        assert not (data_dir / "Hill_of_Towie_ShutdownDuration.zip").exists()
        assert np.allclose(df[DataColumns.shutdown_duration], 0.0)


def _write_metadata_csv(data_dir: Path) -> None:
    rows = [
        "Wind Farm,Turbine Name,Latitude,Longitude,Station ID,Manufacturer,Model,Rated power (kW),"
        "Hub Height (m),Rotor Diameter (m),Country,Commercial Operations Date",
        *[
            f"Hill of Towie,T{i:02d},{57.5 + i / 1000},{-3.08 - i / 1000},{2304509 + i},Siemens,SWT-2.3-VS-82,"
            "2300,59,82,UK,2012-05-12"
            for i in range(1, 22)
        ],
    ]
    # the published file carries a UTF-8 BOM
    (data_dir / "Hill_of_Towie_turbine_metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8-sig")


class TestUnpackLocalMetaData:
    @pytest.mark.parametrize(
        ("scada_index_name", "expected_format"),
        [("TimeStamp_StartFormat", "Start"), ("dtTimeStamp", "End"), ("something_else", "Unknown")],
    )
    def test_returns_wind_up_metadata(
        self, unpack: ModuleType, tmp_path: Path, scada_index_name: str, expected_format: str
    ) -> None:
        _write_metadata_csv(tmp_path)
        df = unpack.unpack_local_meta_data(data_dir=tmp_path, scada_index_name=scada_index_name)
        assert list(df.columns) == ["Name", "Latitude", "Longitude", "TimeZone", "TimeSpanMinutes", "TimeFormat"]
        assert len(df) == 21
        assert df["Name"].tolist() == [f"T{i:02d}" for i in range(1, 22)]
        assert df["Latitude"].iloc[0] == pytest.approx(57.501)
        assert df["Longitude"].iloc[0] == pytest.approx(-3.081)
        assert (df["TimeZone"] == "UTC").all()
        assert (df["TimeSpanMinutes"] == 10).all()
        assert (df["TimeFormat"] == expected_format).all()

    def test_cache_respects_scada_index_name(self, unpack: ModuleType, tmp_path: Path) -> None:
        """A second call with a different scada_index_name must not get the first call's TimeFormat."""
        _write_metadata_csv(tmp_path)
        unpack.unpack_local_meta_data(data_dir=tmp_path)
        df = unpack.unpack_local_meta_data(data_dir=tmp_path, scada_index_name="dtTimeStamp")
        assert (df["TimeFormat"] == "End").all()
