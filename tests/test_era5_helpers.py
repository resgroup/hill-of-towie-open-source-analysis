from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from hot_open import era5_helpers
from hot_open.era5_helpers import _build_era5_df


def _make_mock_response(n_hours: int = 3) -> MagicMock:
    def _make_var(val: float) -> MagicMock:
        mock_var = MagicMock()
        mock_var.ValuesAsNumpy.return_value = np.full(n_hours, val, dtype="float32")
        return mock_var

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1704067200  # 2024-01-01 00:00 UTC
    mock_hourly.TimeEnd.return_value = 1704067200 + 3600 * n_hours
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.side_effect = lambda i: _make_var(float(i))

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly
    return mock_response


class TestBuildEra5Df:
    def test_columns_match_fields(self) -> None:
        fields = ["wind_speed_10m", "wind_direction_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert list(df.columns) == ["wind_speed_10m", "wind_direction_10m"]

    def test_row_count_matches_hours(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(n_hours=5), fields)
        assert len(df) == 5

    def test_index_is_utc_datetimeindex(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df.index.dtype == "datetime64[ns, UTC]"

    def test_timestamp_starts_at_expected_value(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df.index[0] == pd.Timestamp("2024-01-01", tz="UTC")

    def test_field_values_are_propagated(self) -> None:
        fields = ["wind_speed_10m", "wind_direction_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df["wind_speed_10m"].iloc[0] == pytest.approx(0.0)  # index 0
        assert df["wind_direction_10m"].iloc[0] == pytest.approx(1.0)  # index 1


class TestGetEra5HourlyDf:
    def test_hot_wrapper_delegates_with_hot_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_generic(**kwargs: object) -> pd.DataFrame:
            captured.update(kwargs)
            return pd.DataFrame()

        monkeypatch.setattr(era5_helpers, "get_era5_hourly_df", fake_generic)
        era5_helpers.get_hot_era5_hourly_df()
        assert captured == {
            "lat": era5_helpers.HOT_LAT,
            "lon": era5_helpers.HOT_LON,
            "start_date": era5_helpers.HOT_ERA5_START,
            "end_date": era5_helpers.HOT_ERA5_END,
            "fields": era5_helpers.HOT_ERA5_FIELDS,
        }

    def test_end_date_defaults_to_today(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_cache_path(**kwargs: object) -> MagicMock:
            # signature mirrors _era5_cache_path(*, lat, lon, start_date, end_date, fields)
            captured["end_date"] = kwargs["end_date"]
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            return mock_path

        monkeypatch.setattr(era5_helpers, "_era5_cache_path", fake_cache_path)
        monkeypatch.setattr(era5_helpers.pd, "read_parquet", lambda _p: pd.DataFrame())
        era5_helpers.get_era5_hourly_df(lat=1.0, lon=2.0)
        today = pd.Timestamp.now(tz="UTC").normalize().strftime("%Y-%m-%d")
        assert captured["end_date"] == today
