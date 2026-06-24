import os
from pathlib import Path

import pandas as pd
import pytest

import hot_open.fastlog_helpers as flh
from hot_open.fastlog_helpers import (
    SIEMENS_PARKS,
    SIEMENS_TAGS,
    _get_fl_resampled_one_device_one_day,
    _get_raw_df_dict,
    _get_tag_list_from_park_id,
)


class TestSiemensParks:
    def test_default_is_hot_only(self) -> None:
        # Guard against re-leaking private park ids into the public default.
        assert {"HOT"} == SIEMENS_PARKS

    def test_tag_list_for_default_park(self) -> None:
        assert _get_tag_list_from_park_id("HOT") == SIEMENS_TAGS

    def test_tag_list_unknown_park_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _get_tag_list_from_park_id("EXAMPLE")

    def test_tag_list_injected_park(self) -> None:
        assert _get_tag_list_from_park_id("EXAMPLE", siemens_parks={"EXAMPLE"}) == SIEMENS_TAGS

    def test_raw_df_dict_unknown_park_raises(self, tmp_path: object) -> None:
        with pytest.raises(NotImplementedError):
            _get_raw_df_dict(
                park_id="EXAMPLE",
                device_id="X",
                start_dt=pd.Timestamp("2024-01-01"),
                end_dt_excl=pd.Timestamp("2024-01-02"),
                filestore_dir=tmp_path,  # type: ignore[arg-type]
                tags=["ActPower_Value"],
            )

    def test_raw_df_dict_injected_park_no_data(self, tmp_path: object) -> None:
        result = _get_raw_df_dict(
            park_id="EXAMPLE",
            device_id="X",
            start_dt=pd.Timestamp("2024-01-01"),
            end_dt_excl=pd.Timestamp("2024-01-02"),
            filestore_dir=tmp_path,  # type: ignore[arg-type]
            tags=["ActPower_Value"],
            siemens_parks={"EXAMPLE"},
        )
        assert result["ActPower_Value"].empty


PARK = "EXAMPLE"
DEVICE = "123"
DAY = pd.Timestamp("2024-01-01")
DAY_END = pd.Timestamp("2024-01-02")


def _write_source_file(*, filestore: Path, date_str: str, mtime: float) -> Path:
    """Create a dummy raw FL file for one day and stamp its mtime."""
    day_dir = filestore / "FL" / PARK / DEVICE / date_str
    day_dir.mkdir(parents=True, exist_ok=True)
    file = day_dir / f"FL{DEVICE}_Wtc_TDI_ActPower_Value_{date_str.replace('-', '_')}.prq"
    file.write_bytes(b"x")
    os.utime(file, (mtime, mtime))
    return file


def _call(filestore: Path, cache_dir: Path, *, refresh_cache: bool = False) -> pd.DataFrame:
    return _get_fl_resampled_one_device_one_day(
        park_id=PARK,
        device_id=DEVICE,
        start_dt=DAY,
        end_dt_excl=DAY_END,
        filestore_dir=filestore,
        tags=["ActPower_Value"],
        cache_dir=cache_dir,
        siemens_parks={PARK},
        refresh_cache=refresh_cache,
    )


class TestCacheSourceFreshness:
    """Per-day cache self-invalidates when source files are newer than the cached parquet."""

    @pytest.fixture
    def spy_make(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Replace the resampler with a stub that records calls and returns a non-empty frame."""
        calls: list[int] = []

        def stub(*, start_dt: pd.Timestamp, **_: object) -> pd.DataFrame:
            calls.append(1)
            idx = pd.DatetimeIndex([start_dt], name="timestamp")
            return pd.DataFrame({"ActPower_Value": [1.0]}, index=idx)

        monkeypatch.setattr(flh, "make_fl_resampled_one_device", stub)
        return calls

    def test_fresh_cache_is_reused(self, tmp_path: Path, spy_make: list[int]) -> None:
        filestore, cache = tmp_path / "fs", tmp_path / "cache"
        _write_source_file(filestore=filestore, date_str="2024-01-01", mtime=1000.0)
        _call(filestore, cache)  # computes + writes cache (mtime = now > 1000)
        _call(filestore, cache)  # source unchanged -> cache hit
        assert sum(spy_make) == 1

    def test_newer_source_invalidates_cache(self, tmp_path: Path, spy_make: list[int]) -> None:
        filestore, cache = tmp_path / "fs", tmp_path / "cache"
        src = _write_source_file(filestore=filestore, date_str="2024-01-01", mtime=1000.0)
        _call(filestore, cache)
        cache_file = next((cache / "fl_resampled" / PARK / DEVICE).glob("*.parquet"))
        os.utime(src, (cache_file.stat().st_mtime + 1000, cache_file.stat().st_mtime + 1000))
        _call(filestore, cache)  # source now newer than cache -> recompute
        assert sum(spy_make) == 2

    def test_refresh_cache_forces_recompute(self, tmp_path: Path, spy_make: list[int]) -> None:
        filestore, cache = tmp_path / "fs", tmp_path / "cache"
        _write_source_file(filestore=filestore, date_str="2024-01-01", mtime=1000.0)
        _call(filestore, cache)
        _call(filestore, cache, refresh_cache=True)  # fresh cache, but forced
        assert sum(spy_make) == 2

    @pytest.mark.usefixtures("spy_make")
    def test_refresh_cache_overwrites_same_file(self, tmp_path: Path) -> None:
        filestore, cache = tmp_path / "fs", tmp_path / "cache"
        _write_source_file(filestore=filestore, date_str="2024-01-01", mtime=1000.0)
        _call(filestore, cache)
        _call(filestore, cache, refresh_cache=True)
        parquets = list((cache / "fl_resampled" / PARK / DEVICE).glob("*.parquet"))
        assert len(parquets) == 1  # refresh_cache excluded from the cache key

    def test_neighbour_day_backfill_invalidates_cache(self, tmp_path: Path, spy_make: list[int]) -> None:
        filestore, cache = tmp_path / "fs", tmp_path / "cache"
        _write_source_file(filestore=filestore, date_str="2024-01-01", mtime=1000.0)
        _write_source_file(filestore=filestore, date_str="2023-12-31", mtime=1000.0)
        _call(filestore, cache)
        cache_file = next((cache / "fl_resampled" / PARK / DEVICE).glob("*.parquet"))
        prev = filestore / "FL" / PARK / DEVICE / "2023-12-31"
        newer = cache_file.stat().st_mtime + 1000
        for f in prev.iterdir():
            os.utime(f, (newer, newer))
        _call(filestore, cache)  # day-1 (ffill context) backfilled -> recompute
        assert sum(spy_make) == 2
