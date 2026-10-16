from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from wind_up.reanalysis_data import MastOrLiDARDataset

from hot_open.lidar_helpers import (
    LIDAR_AIR_DENSITY_COL,
    add_shear_and_veer,
    calc_air_density_iec,
    calculate_shear,
    calculate_veer,
    extract_data,
    load_hot_lidar_10min_data,
    load_zx_lidar_10min_data,
    load_zx_lidar_fl_data,
)

if TYPE_CHECKING:
    from pathlib import Path

ZX300_ID = "2428"
ZXTM_ID = "5060"
HEIGHTS_M = (38, 58, 78)
DAY = pd.Timestamp("2025-08-22", tz="UTC")


@pytest.fixture
def lidar_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Create an extracted ``lidar_data`` tree under a tmp data dir, so ensure_extracted makes no network call."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("HOT_OPEN_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HOT_OPEN_CACHE_DIR", str(tmp_path / "cache"))
    lidar_dir = data_dir / "lidar_data"
    for unit in (ZX300_ID, ZXTM_ID):
        (lidar_dir / "timeseries" / unit).mkdir(parents=True)
    return lidar_dir


def _lidar_frame(day: pd.Timestamp, *, freq: str, unit: str) -> pd.DataFrame:
    """Build one day of LiDAR data shaped like the published parquet files.

    A naive datetime64 timestamp column, Met columns, per-height speed/direction and a Checksum column.
    """
    stamps = pd.date_range(day.tz_localize(None), periods=pd.Timedelta(days=1) // pd.Timedelta(freq), freq=freq)
    n = len(stamps)
    ramp = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "Met Compass Bearing (deg)": np.full(n, 140.0),
            "Met Air Temp. (C)": np.full(n, 15.0),
            "Met Pressure (mbar)": np.full(n, 1013.25),
            "Met Humidity (%)": np.zeros(n),
        }
    )
    for h in HEIGHTS_M:
        df[f"Horizontal Wind Speed (m/s) at {h}m"] = 8.0 * (h / 58) ** 0.2 + ramp / 1000
        df[f"Horizontal Wind Speed Min (m/s) at {h}m"] = 5.0
        df[f"Wind Direction (deg) at {h}m"] = 200.0 + 0.1 * h
    df["Checksum:abc"] = "x"
    if int(unit) >= 5000:
        df.insert(0, "Timestamp (ISO 8601)", stamps)
    else:
        df.insert(0, "Time and Date", stamps)
        df.insert(1, "Timestamp (s)", (stamps - pd.Timestamp("2000-01-01")).total_seconds().astype("int64"))
    return df


def _ztm_prefix(unit: str) -> str:
    return "ZTM" if int(unit) >= 5000 else ""


def _write_10min(lidar_dir: Path, unit: str, day: pd.Timestamp, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = _lidar_frame(day, freq="10min", unit=unit) if df is None else df
    name = f"Wind10_{_ztm_prefix(unit)}{unit}@{day.strftime('Y%Y_M%m_D%d')}.parquet"
    df.to_parquet(lidar_dir / "timeseries" / unit / name)
    return df


def _write_fl(lidar_dir: Path, unit: str, day: pd.Timestamp, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = _lidar_frame(day, freq="1min", unit=unit) if df is None else df
    name = f"Wind_{_ztm_prefix(unit)}{unit}@{day.strftime('Y%Y_M%m_D%d')}.parquet"
    df.to_parquet(lidar_dir / "timeseries" / unit / name)
    return df


class TestLoadZxLidar10minData:
    @pytest.mark.parametrize("unit", [ZX300_ID, ZXTM_ID])
    def test_reads_window_across_daily_files(self, lidar_dir: Path, unit: str) -> None:
        for offset in (-10, -1, 0, 1, 2, 10):
            _write_10min(lidar_dir, unit, DAY + pd.Timedelta(days=offset))
        start_dt = DAY + pd.Timedelta(hours=12)
        end_dt_excl = DAY + pd.Timedelta(days=1, hours=12)
        df = load_zx_lidar_10min_data(
            data_dir=lidar_dir, lidar_unit_id=unit, start_dt=start_dt, end_dt_excl=end_dt_excl
        )
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"
        assert str(df.index.tz) == "UTC"
        assert df.index.is_monotonic_increasing
        assert df.index.min() == start_dt
        assert df.index.max() == end_dt_excl - pd.Timedelta(minutes=10)
        assert len(df) == 144
        assert not any(c.startswith("Checksum") for c in df.columns)
        assert "Horizontal Wind Speed (m/s) at 58m" in df.columns

    def test_ignores_non_10min_files(self, lidar_dir: Path) -> None:
        _write_10min(lidar_dir, ZX300_ID, DAY)
        _write_fl(lidar_dir, ZX300_ID, DAY)
        df = load_zx_lidar_10min_data(
            data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1)
        )
        assert len(df) == 144

    def test_parses_us_style_string_timestamps(self, lidar_dir: Path) -> None:
        df_in = _lidar_frame(DAY, freq="10min", unit=ZX300_ID)
        df_in["Time and Date"] = df_in["Time and Date"].dt.strftime("%m/%d/%Y %I:%M:%S %p")
        _write_10min(lidar_dir, ZX300_ID, DAY, df_in)
        df = load_zx_lidar_10min_data(
            data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1)
        )
        assert df.index[0] == DAY
        assert df.index[-1] == DAY + pd.Timedelta(hours=23, minutes=50)
        assert all(isinstance(v, str) for v in df["Time and Date"])

    def test_remove_bad_values_replaces_sentinels_with_nan(self, lidar_dir: Path) -> None:
        df_in = _lidar_frame(DAY, freq="10min", unit=ZX300_ID)
        df_in.loc[0, "Horizontal Wind Speed (m/s) at 58m"] = 9999
        df_in.loc[1, "Wind Direction (deg) at 58m"] = 9991
        _write_10min(lidar_dir, ZX300_ID, DAY, df_in)
        kwargs = {
            "data_dir": lidar_dir,
            "lidar_unit_id": ZX300_ID,
            "start_dt": DAY,
            "end_dt_excl": DAY + pd.Timedelta(days=1),
        }
        raw = load_zx_lidar_10min_data(**kwargs)  # type: ignore[arg-type]
        assert raw["Horizontal Wind Speed (m/s) at 58m"].iloc[0] == 9999
        cleaned = load_zx_lidar_10min_data(**kwargs, remove_bad_values=True)  # type: ignore[arg-type]
        assert np.isnan(cleaned["Horizontal Wind Speed (m/s) at 58m"].iloc[0])
        assert np.isnan(cleaned["Wind Direction (deg) at 58m"].iloc[1])
        assert cleaned["Horizontal Wind Speed (m/s) at 58m"].iloc[1:].notna().all()

    def test_no_files_returns_empty_frame(self, lidar_dir: Path) -> None:
        df = load_zx_lidar_10min_data(
            data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1)
        )
        assert df.empty

    def test_file_with_timestamps_from_another_day_raises(self, lidar_dir: Path) -> None:
        """A file whose contents do not match the date in its name is corrupt and must not be loaded silently."""
        df_in = _lidar_frame(DAY + pd.Timedelta(days=5), freq="10min", unit=ZX300_ID)
        _write_10min(lidar_dir, ZX300_ID, DAY, df_in)
        with pytest.raises(ValueError, match="something is wrong"):
            load_zx_lidar_10min_data(
                data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1)
            )


class TestLoadHotLidar10minData:
    def test_returns_both_lidars_with_ztm_bearing_corrected(self, lidar_dir: Path) -> None:
        _write_10min(lidar_dir, ZX300_ID, DAY)
        _write_10min(lidar_dir, ZXTM_ID, DAY)
        datasets = load_hot_lidar_10min_data(data_dir=lidar_dir, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1))
        assert all(isinstance(d, MastOrLiDARDataset) for d in datasets)
        assert [d.id for d in datasets] == ["ZX300_2428", "ZTM_5060"]
        zx300, ztm = datasets
        assert len(zx300.data) == len(ztm.data) == 144
        assert (zx300.data["Met Compass Bearing (deg)"] == 140.0).all()
        assert np.allclose(ztm.data["Met Compass Bearing (deg)"], (140.0 - 131.5) % 360)


class TestCalcAirDensityIec:
    def test_dry_air_at_standard_conditions(self) -> None:
        rho = calc_air_density_iec(temp_c=15.0, pressure_mbar=1013.25, humidity_percent=0.0)
        assert rho == pytest.approx(1.2250, abs=1e-4)

    def test_humidity_lowers_density(self) -> None:
        dry = calc_air_density_iec(temp_c=15.0, pressure_mbar=1013.25, humidity_percent=0.0)
        humid = calc_air_density_iec(temp_c=15.0, pressure_mbar=1013.25, humidity_percent=100.0)
        assert humid < dry
        assert humid == pytest.approx(1.2172, abs=1e-3)

    def test_series_input_returns_series(self) -> None:
        temp = pd.Series([0.0, 15.0, 30.0])
        rho = calc_air_density_iec(
            temp_c=temp, pressure_mbar=pd.Series([1013.25] * 3), humidity_percent=pd.Series([50.0] * 3)
        )
        assert isinstance(rho, pd.Series)
        assert rho.is_monotonic_decreasing


class TestLoadZxLidarFlData:
    @pytest.mark.parametrize("unit", [ZX300_ID, ZXTM_ID])
    def test_reads_window_as_utc(self, lidar_dir: Path, unit: str) -> None:
        for offset in range(2):
            _write_fl(lidar_dir, unit, DAY + pd.Timedelta(days=offset))
        start_dt = DAY + pd.Timedelta(hours=6)
        end_dt_excl = DAY + pd.Timedelta(days=1, hours=6)
        df = load_zx_lidar_fl_data(data_dir=lidar_dir, lidar_unit_id=unit, start_dt=start_dt, end_dt_excl=end_dt_excl)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert str(df.index.tz) == "UTC"
        assert df.index.is_monotonic_increasing
        assert df.index.min() == start_dt
        assert df.index.max() == end_dt_excl - pd.Timedelta(minutes=1)
        assert len(df) == 24 * 60
        assert not any(c.startswith("Checksum") for c in df.columns)

    def test_missing_day_is_skipped_with_warning(self, lidar_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
        _write_fl(lidar_dir, ZX300_ID, DAY)
        with caplog.at_level(logging.WARNING, logger="hot_open.lidar_helpers"):
            df = load_zx_lidar_fl_data(
                data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=2)
            )
        assert len(df) == 24 * 60
        assert "not found" in caplog.text

    def test_no_files_returns_empty_frame(self, lidar_dir: Path) -> None:
        df = load_zx_lidar_fl_data(
            data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1)
        )
        assert df.empty

    def test_duplicate_timestamps_keep_last(self, lidar_dir: Path) -> None:
        df_in = _lidar_frame(DAY, freq="1min", unit=ZX300_ID)
        dup = df_in.iloc[[10]].copy()
        dup["Met Air Temp. (C)"] = 99.0
        _write_fl(lidar_dir, ZX300_ID, DAY, pd.concat([df_in, dup], ignore_index=True))
        df = load_zx_lidar_fl_data(
            data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1)
        )
        assert df.index.is_unique
        assert len(df) == 24 * 60
        assert df["Met Air Temp. (C)"].iloc[10] == 99.0

    def test_remove_bad_values_and_add_air_density(self, lidar_dir: Path) -> None:
        df_in = _lidar_frame(DAY, freq="1min", unit=ZX300_ID)
        df_in.loc[0, "Horizontal Wind Speed (m/s) at 58m"] = 9999
        df_in.loc[1, "Met Pressure (mbar)"] = 500.0  # implausible
        df_in.loc[2, "Met Air Temp. (C)"] = -150.0  # implausible, so density out of range
        _write_fl(lidar_dir, ZX300_ID, DAY, df_in)
        df = load_zx_lidar_fl_data(
            data_dir=lidar_dir,
            lidar_unit_id=ZX300_ID,
            start_dt=DAY,
            end_dt_excl=DAY + pd.Timedelta(days=1),
            remove_bad_values=True,
            add_air_density=True,
        )
        assert np.isnan(df["Horizontal Wind Speed (m/s) at 58m"].iloc[0])
        assert np.isnan(df["Met Pressure (mbar)"].iloc[1])
        assert np.isnan(df[LIDAR_AIR_DENSITY_COL].iloc[1])
        assert np.isnan(df[LIDAR_AIR_DENSITY_COL].iloc[2])
        assert df[LIDAR_AIR_DENSITY_COL].iloc[3:].to_numpy() == pytest.approx(1.2250, abs=1e-4)

    def test_writes_parquet_cache(self, lidar_dir: Path, tmp_path: Path) -> None:
        _write_fl(lidar_dir, ZX300_ID, DAY)
        df = load_zx_lidar_fl_data(
            data_dir=lidar_dir, lidar_unit_id=ZX300_ID, start_dt=DAY, end_dt_excl=DAY + pd.Timedelta(days=1)
        )
        cached = list((tmp_path / "cache").rglob(f"{ZX300_ID}_*.parquet"))
        assert len(cached) == 1
        pd.testing.assert_frame_equal(pd.read_parquet(cached[0]), df)

    @pytest.mark.xfail(
        reason="the cache is written under cache_dir/lidar_raw but looked up under cache_dir/load_zx_lidar_fl_data, "
        "so it is never hit",
        strict=True,
    )
    def test_second_call_is_served_from_cache(self, lidar_dir: Path) -> None:
        """Once loaded, the same request must not need the source files again."""
        _write_fl(lidar_dir, ZX300_ID, DAY)
        kwargs = {
            "data_dir": lidar_dir,
            "lidar_unit_id": ZX300_ID,
            "start_dt": DAY,
            "end_dt_excl": DAY + pd.Timedelta(days=1),
        }
        first = load_zx_lidar_fl_data(**kwargs)  # type: ignore[arg-type]
        for f in (lidar_dir / "timeseries" / ZX300_ID).glob("*.parquet"):
            f.unlink()
        second = load_zx_lidar_fl_data(**kwargs)  # type: ignore[arg-type]
        pd.testing.assert_frame_equal(first, second)


def _profile_frame(ws_rows: list[list[float]], wd_rows: list[list[float]], heights: list[int]) -> pd.DataFrame:
    data: dict[str, list[float]] = {}
    for j, h in enumerate(heights):
        data[f"Horizontal Wind Speed (m/s) at {h}m"] = [r[j] for r in ws_rows]
        data[f"Horizontal Wind Speed Min (m/s) at {h}m"] = [0.0 for _ in ws_rows]
        data[f"Wind Direction (deg) at {h}m"] = [r[j] for r in wd_rows]
    data["Met Wind Direction (deg)"] = [0.0 for _ in ws_rows]
    return pd.DataFrame(data)


class TestExtractData:
    def test_columns_are_heights_sorted_ascending(self) -> None:
        df = _profile_frame([[1.0, 2.0, 3.0]], [[10.0, 20.0, 30.0]], heights=[100, 38, 58])
        ws = extract_data(df, "Horizontal Wind Speed (m/s)")
        assert ws.columns.tolist() == [38.0, 58.0, 100.0]
        assert ws.iloc[0].tolist() == [2.0, 3.0, 1.0]
        wd = extract_data(df, "Wind Direction (deg)")
        assert wd.columns.tolist() == [38.0, 58.0, 100.0]


class TestShearAndVeer:
    def test_power_law_profile_recovers_alpha(self) -> None:
        heights = np.array([40.0, 80.0, 120.0])
        ws = pd.DataFrame([8.0 * (heights / 80) ** 0.2, 6.0 * (heights / 80) ** 0.35], columns=heights)
        alpha, resid = calculate_shear(ws)
        assert alpha == pytest.approx([0.2, 0.35])
        assert resid == pytest.approx([0.0, 0.0], abs=1e-12)

    def test_shear_needs_two_positive_finite_points(self) -> None:
        ws = pd.DataFrame([[5.0, np.nan, 0.0], [5.0, 6.0, np.nan]], columns=[40.0, 80.0, 120.0])
        alpha, resid = calculate_shear(ws)
        assert np.isnan(alpha[0])
        assert np.isnan(resid[0])
        assert alpha[1] == pytest.approx(np.log(6 / 5) / np.log(2))

    def test_linear_veer_across_north_is_unwrapped(self) -> None:
        """Directions straddling 360/0 deg must be fitted as a continuous profile, not a 350 deg jump."""
        wd = pd.DataFrame([[355.0, 5.0, 15.0], [100.0, 90.0, 80.0]], columns=[40.0, 90.0, 140.0])
        slope, r2 = calculate_veer(wd)
        assert slope == pytest.approx([0.2, -0.2])
        assert r2 == pytest.approx([1.0, 1.0])

    def test_veer_constant_direction_has_undefined_r2(self) -> None:
        wd = pd.DataFrame([[200.0, 200.0, 200.0], [np.nan, 200.0, np.nan]], columns=[40.0, 90.0, 140.0])
        slope, r2 = calculate_veer(wd)
        assert slope[0] == pytest.approx(0.0, abs=1e-12)
        assert np.isnan(r2[0])
        assert np.isnan(slope[1])

    def test_add_shear_and_veer_appends_columns(self) -> None:
        heights = [40, 80, 120]
        ws_rows = [[8.0 * (h / 80) ** 0.25 for h in heights]]
        wd_rows = [[200.0, 210.0, 220.0]]
        df = add_shear_and_veer(_profile_frame(ws_rows, wd_rows, heights))
        for col in (
            "Vertical Wind Shear Exponent",
            "Normalised Mean Shear Fit Residuals",
            "Vertical Wind Veer",
            "Vertical Wind Veer R Squared",
        ):
            assert col in df.columns
        assert df["Vertical Wind Shear Exponent"].iloc[0] == pytest.approx(0.25)
        assert df["Vertical Wind Veer"].iloc[0] == pytest.approx(0.25)
        assert df["Vertical Wind Veer R Squared"].iloc[0] == pytest.approx(1.0)
