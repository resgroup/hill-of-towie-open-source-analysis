import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from hot_open.era5_helpers import _build_era5_df


def _make_mock_response(n_hours: int = 3) -> MagicMock:
    mock_var = MagicMock()
    mock_var.ValuesAsNumpy.return_value = np.ones(n_hours, dtype="float32")

    mock_hourly = MagicMock()
    mock_hourly.Time.return_value = 1704067200  # 2024-01-01 00:00 UTC
    mock_hourly.TimeEnd.return_value = 1704067200 + 3600 * n_hours
    mock_hourly.Interval.return_value = 3600
    mock_hourly.Variables.return_value = mock_var

    mock_response = MagicMock()
    mock_response.Hourly.return_value = mock_hourly
    return mock_response


class TestBuildEra5Df:
    def test_columns_match_fields(self) -> None:
        fields = ["wind_speed_10m", "wind_direction_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert list(df.columns) == ["timestamp", "wind_speed_10m", "wind_direction_10m"]

    def test_row_count_matches_hours(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(n_hours=5), fields)
        assert len(df) == 5

    def test_timestamp_is_utc(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df["timestamp"].dtype == "datetime64[ns, UTC]"

    def test_timestamp_starts_at_expected_value(self) -> None:
        fields = ["wind_speed_10m"]
        df = _build_era5_df(_make_mock_response(), fields)
        assert df["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz="UTC")
