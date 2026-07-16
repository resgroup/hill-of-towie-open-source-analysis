import os
from pathlib import Path

import pandas as pd
import pytest

import hot_open.fastlog_helpers as flh
from hot_open.fastlog_helpers import (
    SIEMENS_PARKS,
    SIEMENS_TAGS,
    TIMESTAMP_NAME,
    _get_fl_resampled_one_device_one_day,
    _get_raw_df_dict,
    _get_tag_list_from_park_id,
    resample_fastlog_tags,
    upsample_and_ffill_stopping_at_nans,
)


class TestSiemensParks:
    def test_default_is_hot_only(self) -> None:
        # Guard against leaking private park ids into the public default.
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

    def test_refresh_cache_removes_stale_cache_on_empty_recompute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filestore, cache = tmp_path / "fs", tmp_path / "cache"
        _write_source_file(filestore=filestore, date_str="2024-01-01", mtime=1000.0)
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01")], name="timestamp")
        monkeypatch.setattr(
            flh, "make_fl_resampled_one_device", lambda **_: pd.DataFrame({"ActPower_Value": [1.0]}, index=idx)
        )
        _call(filestore, cache)
        parquet = next((cache / "fl_resampled" / PARK / DEVICE).glob("*.parquet"))
        # Recompute now yields an empty frame (e.g. source removed/filtered): the stale parquet
        # must not survive, or a later non-refresh run would silently reuse it.
        monkeypatch.setattr(flh, "make_fl_resampled_one_device", lambda **_: pd.DataFrame())
        _call(filestore, cache, refresh_cache=True)
        assert not parquet.exists()

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


def _latched_tag_df(times_values: list[tuple[str, bool]]) -> pd.DataFrame:
    """Build a single-column latched (boolean, logged-only-on-change) tag DataFrame."""
    idx = pd.DatetimeIndex([pd.Timestamp(t) for t, _ in times_values], name=TIMESTAMP_NAME)
    return pd.DataFrame({"alarms_TEST": [v for _, v in times_values]}, index=idx)


class TestTrailingObservationPreserved:
    """A latched value observed in the final partial sub-timebase cell must not be dropped.

    Regression for the FL resample forward-fill bug: ``resample(rule).ffill()`` builds its grid
    only out to ``floor(last_observation)`` and samples each grid point as-of the *previous*
    observation, so an alarm that clears at e.g. 16:40:19.4 (the last row of the day) never lands
    on the upsampled grid. The cleared value was lost, the minute read as still-raised, and the
    downstream plot forward-filled the stuck value to the end of the day.
    """

    # Mirrors the shape of a real alarm that fired then cleared, with the final clear at a
    # sub-second time that is the last row for the tag.
    FIRE = "2026-07-15 13:20:18.598"
    CLEAR = "2026-07-15 16:40:19.428"

    def test_upsample_keeps_final_cleared_value(self) -> None:
        tag_df = _latched_tag_df([(self.FIRE, True), (self.CLEAR, False)])
        up = upsample_and_ffill_stopping_at_nans(
            tag_df=tag_df,
            timebase_s=60,
            subsampling_timebase_ms=1000,
            only_ffill_one_timebase=False,
        )
        # The final observed value (cleared) must be represented, at its own timestamp (not shifted).
        assert up.index[-1] == pd.Timestamp(self.CLEAR)
        assert bool(up["alarms_TEST"].dropna().iloc[-1]) is False

    def test_final_minute_reads_cleared_after_coarse_resample(self) -> None:
        # Full assemble-and-resample path: the 16:40 minute must read cleared (False), not raised.
        tag_df = _latched_tag_df([(self.FIRE, True), (self.CLEAR, False)])
        resampled = resample_fastlog_tags(raw_df_dict={"alarms_TEST": tag_df}, timebase_s=60)
        col = resampled["alarms_TEST"]
        assert bool(col.loc["2026-07-15 16:39:00"]) is True
        assert bool(col.loc["2026-07-15 16:40:00"]) is False

    def test_final_observation_stays_in_its_own_coarse_bin(self) -> None:
        # A clear late in the final minute (…:59.9) must resample into THAT minute, not be rounded
        # forward into the next one (nor create a spurious extra bin). Guards the reviewer's concern
        # about anchoring the re-appended point at ceil(freq) instead of the observation time.
        tag_df = _latched_tag_df([("2026-07-15 16:00:00", True), ("2026-07-15 16:40:59.900", False)])
        col = resample_fastlog_tags(raw_df_dict={"alarms_TEST": tag_df}, timebase_s=60)["alarms_TEST"]
        assert bool(col.loc["2026-07-15 16:40:00"]) is False
        assert pd.Timestamp("2026-07-15 16:41:00") not in col.index


# A small committed slice of real Hill of Towie fastlog (turbine 2304510, 2026-01-02, ~08:05-08:09,
# spanning a real GeneratOut->LargeOnGrd stop/restart). The canonical Zenodo dataset is a single
# multi-GB zip, too large to download in CI, so this trimmed sample is committed under test_data.
TEST_DATA_DIR = Path(__file__).parent / "test_data"
FL_SAMPLE_DIR = TEST_DATA_DIR / "fl_sample"
FL_SAMPLE_TAGS = ["ActPower_Value", "AcWindSp_AcWindSp", "GenState_GenState"]


class TestResampleRealHotSample:
    """Resampling real HOT fastlog through the public path, with the sparse latched GenState tag."""

    def _resampled(self) -> pd.DataFrame:
        raw = _get_raw_df_dict(
            park_id="HOT",
            device_id="2304510",
            start_dt=pd.Timestamp("2026-01-02 08:06:00"),
            # Ends just after the 08:07:40.640 LargeOnGrd observation so that value is the final
            # row read for GenState -- the exact shape that triggered the dropped-observation bug.
            end_dt_excl=pd.Timestamp("2026-01-02 08:07:41"),
            filestore_dir=FL_SAMPLE_DIR,
            tags=FL_SAMPLE_TAGS,
        )
        return resample_fastlog_tags(raw_df_dict=raw, timebase_s=60)

    def test_sparse_tag_final_observation_survives(self) -> None:
        # The generator returned (LargeOnGrd) at 08:07:40.640, the last GenState row in the window.
        # Before the fix this sub-second final observation was dropped and 08:07 read "GeneratOut".
        gen = self._resampled()["GenState_GenState"]
        assert gen.loc["2026-01-02 08:06:00"] == "GeneratOut"
        assert gen.loc["2026-01-02 08:07:00"] == "LargeOnGrd"

    def test_matches_committed_characterization(self) -> None:
        expected = pd.read_parquet(TEST_DATA_DIR / "fl_sample_resampled_expected.parquet")
        pd.testing.assert_frame_equal(self._resampled(), expected, check_exact=False, check_freq=False)


class TestUpsampleUnchangedExceptTrailingPoint:
    """The trailing-observation fix must not alter normal resampling; it only appends a tail point.

    On real HOT fastlog, ``upsample_and_ffill_stopping_at_nans`` must equal the historical
    ``resample(freq).ffill()`` on every row they share and add at most one trailing grid point. This
    holds -- and these assertions pass -- both before and after the fix (before, the two are identical
    and no point is added; after, only the final observation's grid point is appended). It is the
    guard that the fix left resampling of data *without* a dropped dangling record byte-for-byte intact.
    """

    def _raw(self, tag: str) -> pd.DataFrame:
        raw = _get_raw_df_dict(
            park_id="HOT",
            device_id="2304510",
            start_dt=pd.Timestamp("2026-01-02 08:06:00"),
            end_dt_excl=pd.Timestamp("2026-01-02 08:07:41"),
            filestore_dir=FL_SAMPLE_DIR,
            tags=[tag],
        )
        return raw[tag]

    @pytest.mark.parametrize(
        ("tag", "only_ffill_one_timebase"),
        [
            ("ActPower_Value", True),  # busy numeric tag: limited (one-timebase) ffill
            ("GenState_GenState", False),  # sparse latched tag: unlimited ffill
        ],
    )
    def test_only_appends_trailing_point(self, tag: str, only_ffill_one_timebase: bool) -> None:  # noqa: FBT001
        tag_df = self._raw(tag)
        timebase_s, subsampling_timebase_ms = 60, 1000
        ffill_limit = None if not only_ffill_one_timebase else timebase_s * 1000 // subsampling_timebase_ms - 1
        # The historical behaviour the fix must preserve everywhere it overlaps.
        old = tag_df.resample(pd.Timedelta(milliseconds=subsampling_timebase_ms)).ffill(limit=ffill_limit)

        new = upsample_and_ffill_stopping_at_nans(
            tag_df=tag_df,
            timebase_s=timebase_s,
            subsampling_timebase_ms=subsampling_timebase_ms,
            only_ffill_one_timebase=only_ffill_one_timebase,
        )

        # No pre-existing grid point is changed, and at most one trailing point is added.
        pd.testing.assert_frame_equal(new.loc[old.index], old, check_freq=False)
        assert 0 <= len(new) - len(old) <= 1
