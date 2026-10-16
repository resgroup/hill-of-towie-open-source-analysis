from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

import flatbuffers  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import pytest
import responses
from wind_up.reanalysis_data import ReanalysisDataset

from hot_open import era5_helpers
from hot_open.era5_helpers import _build_era5_df

if TYPE_CHECKING:
    from pathlib import Path

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def _open_meteo_body(*, start_s: int, n_hours: int, values: list[np.ndarray]) -> bytes:
    """Encode a size-prefixed Open-Meteo WeatherApiResponse flatbuffer carrying only hourly data.

    The SDK ships readers but no builders, so the vtable slots are written by hand; slot numbers
    are the reader's vtable offsets (4, 6, 8, ...) mapped to 0, 1, 2, ...
    """
    builder = flatbuffers.Builder(1024)
    var_offsets = []
    for vals in values:
        vec = builder.CreateNumpyVector(np.asarray(vals, dtype="float32"))
        builder.StartObject(4)  # VariableWithValues
        builder.PrependUOffsetTRelativeSlot(3, vec, 0)  # values
        var_offsets.append(builder.EndObject())
    builder.StartVector(4, len(var_offsets), 4)
    for off in reversed(var_offsets):
        builder.PrependUOffsetTRelative(off)
    variables = builder.EndVector()
    builder.StartObject(4)  # VariablesWithTime
    builder.PrependInt64Slot(0, start_s, 0)  # time
    builder.PrependInt64Slot(1, start_s + 3600 * n_hours, 0)  # time_end
    builder.PrependInt32Slot(2, 3600, 0)  # interval
    builder.PrependUOffsetTRelativeSlot(3, variables, 0)  # variables
    hourly = builder.EndObject()
    builder.StartObject(12)  # WeatherApiResponse
    builder.PrependUOffsetTRelativeSlot(11, hourly, 0)  # hourly
    builder.FinishSizePrefixed(builder.EndObject())
    return bytes(builder.Output())


@pytest.fixture
def era5_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Point the ERA5 parquet cache and the HTTP request cache at tmp_path."""
    cache_dir = tmp_path / "era5_data"
    cache_dir.mkdir()
    monkeypatch.setattr(era5_helpers, "_era5_cache_dir", cache_dir)
    monkeypatch.setenv("HOT_OPEN_CACHE_DIR", str(tmp_path / "cache"))
    return cache_dir


class TestGetEra5HourlyDfFetch:
    @responses.activate
    def test_fetches_decodes_and_caches(self, era5_cache: Path) -> None:
        """The full request path, from HTTP through the flatbuffer decode to the parquet cache."""
        n_hours = 4
        ws = np.array([5.0, 6.0, 7.0, 8.0])
        wd = np.array([180.0, 190.0, 200.0, 210.0])
        responses.add(
            responses.GET,
            ARCHIVE_URL,
            body=_open_meteo_body(start_s=1704067200, n_hours=n_hours, values=[ws, wd]),
            content_type="application/octet-stream",
        )
        fields = ["wind_speed_100m", "wind_direction_100m"]
        kwargs = {"lat": 57.5, "lon": -3.25, "start_date": "2024-01-01", "end_date": "2024-01-01", "fields": fields}

        df = era5_helpers.get_era5_hourly_df(**kwargs)  # type: ignore[arg-type]

        assert list(df.columns) == fields
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"
        assert str(df.index.tz) == "UTC"
        assert df.index.tolist() == list(pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC"))
        assert df["wind_speed_100m"].to_numpy() == pytest.approx(ws)
        assert df["wind_direction_100m"].to_numpy() == pytest.approx(wd)

        assert len(responses.calls) == 1
        params = parse_qs(urlparse(str(responses.calls[0].request.url)).query)
        assert params["latitude"] == ["57.5"]
        assert params["longitude"] == ["-3.25"]
        assert params["hourly"] == fields
        assert params["models"] == ["era5"]
        assert params["wind_speed_unit"] == ["ms"]
        assert params["format"] == ["flatbuffers"]
        assert len(list(era5_cache.glob("ERA5_57.50_-3.25_2024-01-01_2024-01-01_*.parquet"))) == 1

        again = era5_helpers.get_era5_hourly_df(**kwargs)  # type: ignore[arg-type]
        assert len(responses.calls) == 1
        pd.testing.assert_frame_equal(again, df)

    @responses.activate
    def test_api_error_raises_and_caches_nothing(self, era5_cache: Path) -> None:
        responses.add(responses.GET, ARCHIVE_URL, status=400, json={"error": True, "reason": "bad request"})
        with pytest.raises(Exception, match="bad request"):
            era5_helpers.get_era5_hourly_df(lat=1.0, lon=2.0, start_date="2024-01-01", end_date="2024-01-02")
        assert not list(era5_cache.glob("*.parquet"))


class TestEra5CachePath:
    def test_is_deterministic_and_argument_sensitive(self, era5_cache: Path) -> None:
        base = {"lat": 57.5, "lon": -3.25, "start_date": "2024-01-01", "end_date": "2024-02-01", "fields": ["a"]}
        p1 = era5_helpers._era5_cache_path(**base)  # type: ignore[arg-type]  # noqa: SLF001
        assert p1 == era5_helpers._era5_cache_path(**base)  # type: ignore[arg-type]  # noqa: SLF001
        assert p1.parent == era5_cache
        assert p1.suffix == ".parquet"
        assert p1 != era5_helpers._era5_cache_path(**{**base, "fields": ["a", "b"]})  # type: ignore[arg-type]  # noqa: SLF001
        assert p1 != era5_helpers._era5_cache_path(**{**base, "lat": 57.51})  # type: ignore[arg-type]  # noqa: SLF001


def test_get_hot_reanalysis_datasets_wraps_hot_era5(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dataset id is what wind-up configs refer to, so a change to it breaks them silently."""
    df = pd.DataFrame({"wind_speed_100m": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"))
    monkeypatch.setattr(era5_helpers, "get_hot_era5_hourly_df", lambda: df)
    datasets = era5_helpers.get_hot_reanalysis_datasets()
    assert len(datasets) == 1
    assert isinstance(datasets[0], ReanalysisDataset)
    assert datasets[0].id == "ERA5_57.50_-3.25"
    assert datasets[0].data is df


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
