import os
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import hot_open.fastlog_helpers as flh
from hot_open.fastlog_helpers import (
    SIEMENS_PARKS,
    SIEMENS_TAGS,
    TIMESTAMP_NAME,
    _get_raw_df_dict,
    _get_tag_list_from_park_id,
    get_fl_resampled_one_device,
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
    """Resample one day through the production entry point, cache and all."""
    return get_fl_resampled_one_device(
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

        monkeypatch.setattr(flh, "_resample_one_chunk", stub)
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
        monkeypatch.setattr(flh, "_resample_one_chunk", lambda **_: pd.DataFrame({"ActPower_Value": [1.0]}, index=idx))
        _call(filestore, cache)
        parquet = next((cache / "fl_resampled" / PARK / DEVICE).glob("*.parquet"))
        # Recompute now yields an empty frame (e.g. source removed/filtered): the stale parquet
        # must not survive, or a later non-refresh run would silently reuse it.
        monkeypatch.setattr(flh, "_resample_one_chunk", lambda **_: pd.DataFrame())
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


def _slow_scada_tag_df(
    tag: str,
    *,
    start: str = "2024-03-01 10:00:00",
    minutes: int = 10,
    period_s: int = 10,
    drop_between: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Build a tag sampled every ``period_s`` (OPC-like), optionally with samples missing.

    Unlike fastlog, a busy tag here reports every ~10s rather than every timebase, so at a fine
    ``timebase_s`` most sub-grid cells hold no sample.
    """
    idx = pd.date_range(
        pd.Timestamp(start),
        pd.Timestamp(start) + pd.Timedelta(minutes=minutes),
        freq=f"{period_s}s",
        name=TIMESTAMP_NAME,
    )
    if drop_between is not None:
        lo, hi = (pd.Timestamp(x) for x in drop_between)
        idx = idx[(idx < lo) | (idx > hi)]
    return pd.DataFrame({tag: range(len(idx))}, index=idx, dtype=float)


class TestBusyTagFfillLimit:
    """``busy_tag_ffill_limit_s`` decouples a busy tag's hold horizon from the output timebase.

    Holding busy tags for one timebase suits fastlog, where they report every timebase. On slower
    logging such as OPC it does not: 10s data on a 1s grid leaves the tag absent from 9 cells in
    10, so every window reads as an outage.
    """

    BUSY = ("busy_a", "busy_b")

    def _raw(self) -> dict[str, pd.DataFrame]:
        return {
            "busy_a": _slow_scada_tag_df("busy_a"),
            "busy_b": _slow_scada_tag_df("busy_b"),
            "slow_c": _slow_scada_tag_df("slow_c", period_s=60),
        }

    def test_one_second_grid_is_unusable_without_a_limit(self) -> None:
        # Guards the status quo this parameter exists to fix (and that the default keeps it).
        resampled = resample_fastlog_tags(raw_df_dict=self._raw(), timebase_s=1, busy_tags=self.BUSY)
        assert resampled["busy_a"].isna().mean() > 0.8

    def test_limit_restores_busy_tag_coverage_on_a_one_second_grid(self) -> None:
        resampled = resample_fastlog_tags(
            raw_df_dict=self._raw(), timebase_s=1, busy_tags=self.BUSY, busy_tag_ffill_limit_s=45
        )
        assert resampled["busy_a"].isna().mean() < 0.05

    def test_limit_equal_to_timebase_reproduces_the_default(self) -> None:
        # The parameter is expressed in the same units as the behaviour it generalises, so setting
        # it to one timebase must be a no-op. This is what makes it safe to default to None.
        default = resample_fastlog_tags(raw_df_dict=self._raw(), timebase_s=60, busy_tags=self.BUSY)
        explicit = resample_fastlog_tags(
            raw_df_dict=self._raw(), timebase_s=60, busy_tags=self.BUSY, busy_tag_ffill_limit_s=60
        )
        pd.testing.assert_frame_equal(default, explicit)

    def test_limit_finer_than_the_subsampling_grid_raises(self) -> None:
        with pytest.raises(ValueError, match="busy_tag_ffill_limit_s"):
            resample_fastlog_tags(
                raw_df_dict=self._raw(), timebase_s=600, busy_tags=self.BUSY, busy_tag_ffill_limit_s=0.5
            )


class TestRequireAllBusyTags:
    """``require_all_busy_tags`` treats busy tags as independent rather than interchangeable.

    The default assumes busy tags fail together. They don't always: a vane channel can die for days
    while power and wind speed keep logging, and the default then forward-fills every other tag
    across the dead stretch.
    """

    BUSY = ("busy_a", "busy_b")
    DEAD = ("2024-03-01 10:02:00", "2024-03-01 10:08:00")
    # Well inside the dead stretch, clear of the one-timebase hold at its leading edge.
    DURING_DEAD = "2024-03-01 10:06:00"

    def _raw(self, *, healthy: bool = False) -> dict[str, pd.DataFrame]:
        return {
            "busy_a": _slow_scada_tag_df("busy_a"),
            "busy_b": _slow_scada_tag_df("busy_b", drop_between=None if healthy else self.DEAD),
            "slow_c": _slow_scada_tag_df("slow_c", period_s=300),
        }

    def test_default_forward_fills_across_one_dead_busy_tag(self) -> None:
        # Status quo: busy_a is still reporting, so nothing is treated as an outage.
        resampled = resample_fastlog_tags(raw_df_dict=self._raw(), timebase_s=60, busy_tags=self.BUSY)
        assert not pd.isna(resampled["slow_c"].loc[self.DURING_DEAD])

    def test_require_all_stops_the_fill_across_one_dead_busy_tag(self) -> None:
        resampled = resample_fastlog_tags(
            raw_df_dict=self._raw(), timebase_s=60, busy_tags=self.BUSY, require_all_busy_tags=True
        )
        assert pd.isna(resampled["slow_c"].loc[self.DURING_DEAD])

    def test_no_effect_when_every_busy_tag_is_reporting(self) -> None:
        # The flag must cost nothing on healthy data, or it is not safe to turn on by default
        # for a whole dataset.
        raw = self._raw(healthy=True)
        pd.testing.assert_frame_equal(
            resample_fastlog_tags(raw_df_dict=raw, timebase_s=60, busy_tags=self.BUSY),
            resample_fastlog_tags(raw_df_dict=raw, timebase_s=60, busy_tags=self.BUSY, require_all_busy_tags=True),
        )


class TestRequireAllBusyTagsWithNoData:
    """A required busy tag with *no* data at all must still count as missing.

    The busy-tag loop skips tags absent from ``raw_df_dict`` or holding an empty frame, so they
    never become a column in the availability mask and would be silently treated as not required --
    the worst case, and a live one, since per-day loading gives an empty frame for any day falling
    wholly inside a dead-channel stretch.
    """

    BUSY = ("busy_a", "busy_b")
    # slow_c samples at 10:00/10:05/10:10, so this point is purely forward-filled. Real
    # observations are deliberately kept even here -- the mask stops bridging, it does not
    # delete data a tag actually reported.
    DURING = "2024-03-01 10:07:00"

    def _raw(self, busy_b: str) -> dict[str, pd.DataFrame]:
        raw = {"busy_a": _slow_scada_tag_df("busy_a"), "slow_c": _slow_scada_tag_df("slow_c", period_s=300)}
        if busy_b == "empty":
            raw["busy_b"] = _slow_scada_tag_df("busy_b").iloc[:0]
        elif busy_b == "present":
            raw["busy_b"] = _slow_scada_tag_df("busy_b")
        return raw

    @pytest.mark.parametrize("busy_b", ["empty", "absent"])
    def test_busy_tag_with_no_data_is_still_required(self, busy_b: str) -> None:
        resampled = resample_fastlog_tags(
            raw_df_dict=self._raw(busy_b), timebase_s=60, busy_tags=self.BUSY, require_all_busy_tags=True
        )
        assert pd.isna(resampled["slow_c"].loc[self.DURING])

    def test_every_busy_tag_missing_masks_everything(self) -> None:
        raw = {"slow_c": _slow_scada_tag_df("slow_c", period_s=300)}
        resampled = resample_fastlog_tags(
            raw_df_dict=raw, timebase_s=60, busy_tags=self.BUSY, require_all_busy_tags=True
        )
        assert pd.isna(resampled["slow_c"].loc[self.DURING])

    def test_default_is_unaffected_by_a_missing_busy_tag(self) -> None:
        resampled = resample_fastlog_tags(raw_df_dict=self._raw("absent"), timebase_s=60, busy_tags=self.BUSY)
        assert not pd.isna(resampled["slow_c"].loc[self.DURING])


class TestBusyTagHorizonIsScopedToBusyTags:
    """``busy_tag_ffill_limit_s`` must reach busy tags only, not everything outside ``ffill_tags``.

    Those two sets coincide only under the default ``ffill_tags``. With a custom list there is a
    third category -- neither busy nor ffill -- which keeps the one-timebase limit; widening it
    would change unrelated output values and contradict the parameter's documented scope.
    """

    def _raw(self) -> dict[str, pd.DataFrame]:
        return {"busy_a": _slow_scada_tag_df("busy_a"), "status_d": _slow_scada_tag_df("status_d", period_s=120)}

    def test_non_busy_non_ffill_tag_keeps_the_one_timebase_limit(self) -> None:
        default = resample_fastlog_tags(raw_df_dict=self._raw(), timebase_s=60, busy_tags=("busy_a",), ffill_tags=())
        widened = resample_fastlog_tags(
            raw_df_dict=self._raw(), timebase_s=60, busy_tags=("busy_a",), ffill_tags=(), busy_tag_ffill_limit_s=600
        )
        pd.testing.assert_series_equal(default["status_d"], widened["status_d"])

    def test_non_busy_min_data_count_tag_keeps_the_one_timebase_limit(self) -> None:
        # A longer horizon would inflate the tag's per-window sample count and stop min_data_count
        # masking the low-coverage windows it is there to catch.
        default = resample_fastlog_tags(
            raw_df_dict=self._raw(),
            timebase_s=60,
            busy_tags=("busy_a",),
            ffill_tags=(),
            min_data_count=30,
            min_data_count_tag="status_d",
        )
        widened = resample_fastlog_tags(
            raw_df_dict=self._raw(),
            timebase_s=60,
            busy_tags=("busy_a",),
            ffill_tags=(),
            min_data_count=30,
            min_data_count_tag="status_d",
            busy_tag_ffill_limit_s=600,
        )
        # busy_a legitimately differs (it *is* busy); status_d and the masking it drives must not.
        pd.testing.assert_series_equal(default["status_d"], widened["status_d"])


def _direction_tag_df(
    tag: str,
    values: list[float],
    *,
    start: str = "2024-03-01 10:00:00",
    period_s: int = 10,
) -> pd.DataFrame:
    """Build a direction tag whose samples are exactly ``values``, one every ``period_s``."""
    idx = pd.date_range(pd.Timestamp(start), periods=len(values), freq=f"{period_s}s", name=TIMESTAMP_NAME)
    return pd.DataFrame({tag: values}, index=idx, dtype=float)


class TestStdTags:
    """``std_tags`` emits a per-window standard deviation, circular-aware.

    Callers who want variability alongside the mean had to compute it themselves, and a plain
    std is wrong for a direction: the spread of 359 and 1 degrees is 1 degree, not 253.
    """

    def test_non_circular_std_matches_pandas_std_on_the_same_grid(self) -> None:
        raw = {"linear_a": _slow_scada_tag_df("linear_a", period_s=10, minutes=5)}
        resampled = resample_fastlog_tags(
            raw_df_dict=raw, timebase_s=60, busy_tags=("linear_a",), std_tags=("linear_a",)
        )
        upsampled = upsample_and_ffill_stopping_at_nans(
            tag_df=raw["linear_a"], timebase_s=60, subsampling_timebase_ms=1000, only_ffill_one_timebase=True
        )
        expected = upsampled["linear_a"].resample("60s").std()
        pd.testing.assert_series_equal(
            resampled["std_linear_a"].dropna(), expected.reindex(resampled.index).dropna(), check_names=False
        )

    def test_std_column_is_named_with_a_prefix(self) -> None:
        raw = {"linear_a": _slow_scada_tag_df("linear_a")}
        resampled = resample_fastlog_tags(
            raw_df_dict=raw, timebase_s=60, busy_tags=("linear_a",), std_tags=("linear_a",)
        )
        assert "std_linear_a" in resampled.columns
        assert "linear_a" in resampled.columns

    def test_no_std_columns_by_default(self) -> None:
        # The parameter must cost nothing when unused, or it is not safe to add.
        raw = {"linear_a": _slow_scada_tag_df("linear_a")}
        default = resample_fastlog_tags(raw_df_dict=raw, timebase_s=60, busy_tags=("linear_a",))
        assert not [c for c in default.columns if c.startswith("std_")]

    def test_circular_std_is_small_for_a_tight_cluster_and_large_for_a_spread(self) -> None:
        tight = _direction_tag_df("AcWindDr_Value", [359.0, 0.0, 1.0, 2.0, 358.0, 0.5])
        spread = _direction_tag_df("AcWindDr_Value", [10.0, 100.0, 190.0, 280.0, 55.0, 145.0])
        kwargs: dict[str, Any] = {
            "timebase_s": 60,
            "busy_tags": ("AcWindDr_Value",),
            "circular_tags": ("AcWindDr_Value",),
            "std_tags": ("AcWindDr_Value",),
        }
        tight_std = resample_fastlog_tags(raw_df_dict={"AcWindDr_Value": tight}, **kwargs)["std_AcWindDr_Value"]
        spread_std = resample_fastlog_tags(raw_df_dict={"AcWindDr_Value": spread}, **kwargs)["std_AcWindDr_Value"]
        # A plain std would call the tight cluster ~180 deg because it straddles 0/360.
        assert tight_std.dropna().max() < 5
        assert spread_std.dropna().min() > 40

    @pytest.mark.parametrize("offset", [0.0, 37.0, 180.0, 359.0])
    def test_circular_std_is_rotation_invariant(self, offset: float) -> None:
        # The property that distinguishes a circular statistic from a linear one.
        values = [10.0, 25.0, 3.0, 355.0, 18.0, 340.0]
        kwargs: dict[str, Any] = {
            "timebase_s": 60,
            "busy_tags": ("AcWindDr_Value",),
            "circular_tags": ("AcWindDr_Value",),
            "std_tags": ("AcWindDr_Value",),
        }
        base = resample_fastlog_tags(
            raw_df_dict={"AcWindDr_Value": _direction_tag_df("AcWindDr_Value", values)}, **kwargs
        )["std_AcWindDr_Value"]
        rotated_values = [(v + offset) % 360 for v in values]
        rotated = resample_fastlog_tags(
            raw_df_dict={"AcWindDr_Value": _direction_tag_df("AcWindDr_Value", rotated_values)}, **kwargs
        )["std_AcWindDr_Value"]
        pd.testing.assert_series_equal(base, rotated, check_names=False)


class TestMinRawDataCount:
    """``min_raw_data_count`` counts *raw* samples, which ``min_data_count`` cannot.

    ``min_data_count`` counts sub-grid cells, and those saturate once ``busy_tag_ffill_limit_s``
    is set: a 60s window starved from 5 raw samples to 1-2 still reports 60 filled cells, so it
    is indistinguishable from a healthy one. That is exactly the case a minimum-coverage rule
    exists to catch, and the busy-tag fill hides it by design.
    """

    BUSY = ("busy_a",)

    def _raw(self, *, period_s: int) -> dict[str, pd.DataFrame]:
        return {"busy_a": _slow_scada_tag_df("busy_a", period_s=period_s, minutes=10)}

    def test_min_data_count_cannot_tell_a_starved_window_from_a_healthy_one(self) -> None:
        # Guards the limitation this parameter exists for; if this ever fails, re-read the docs.
        healthy = resample_fastlog_tags(
            raw_df_dict=self._raw(period_s=12),
            timebase_s=60,
            busy_tags=self.BUSY,
            busy_tag_ffill_limit_s=45,
            min_data_count=4,
        )
        starved = resample_fastlog_tags(
            raw_df_dict=self._raw(period_s=36),
            timebase_s=60,
            busy_tags=self.BUSY,
            busy_tag_ffill_limit_s=45,
            min_data_count=4,
        )
        # Excluding the trailing partial window, which is legitimately short of samples.
        assert healthy["busy_a"].iloc[:-1].notna().all()
        assert starved["busy_a"].iloc[:-1].notna().all()

    def test_masks_a_starved_window(self) -> None:
        starved = resample_fastlog_tags(
            raw_df_dict=self._raw(period_s=36),
            timebase_s=60,
            busy_tags=self.BUSY,
            busy_tag_ffill_limit_s=45,
            min_raw_data_count=4,
        )
        assert starved["busy_a"].isna().all()

    def test_keeps_a_window_meeting_the_threshold(self) -> None:
        healthy = resample_fastlog_tags(
            raw_df_dict=self._raw(period_s=12),
            timebase_s=60,
            busy_tags=self.BUSY,
            busy_tag_ffill_limit_s=45,
            min_raw_data_count=4,
        )
        # 60s / 12s = 5 raw samples per window, so a threshold of 4 must not bite.
        assert healthy["busy_a"].iloc[:-1].notna().all()

    def test_none_by_default_changes_nothing(self) -> None:
        raw = self._raw(period_s=36)
        kwargs: dict[str, Any] = {"timebase_s": 60, "busy_tags": self.BUSY, "busy_tag_ffill_limit_s": 45}
        pd.testing.assert_frame_equal(
            resample_fastlog_tags(raw_df_dict=raw, **kwargs),
            resample_fastlog_tags(raw_df_dict=raw, min_raw_data_count=None, **kwargs),
        )

    def test_polarity_follows_require_all_busy_tags(self) -> None:
        # One busy tag well fed, one starved. require_all -> any tag below threshold masks.
        raw = {
            "busy_a": _slow_scada_tag_df("busy_a", period_s=12, minutes=10),
            "busy_b": _slow_scada_tag_df("busy_b", period_s=60, minutes=10),
        }
        kwargs: dict[str, Any] = {
            "raw_df_dict": raw,
            "timebase_s": 60,
            "busy_tags": ("busy_a", "busy_b"),
            "busy_tag_ffill_limit_s": 45,
            "min_raw_data_count": 4,
        }
        any_below = resample_fastlog_tags(**kwargs, require_all_busy_tags=True)
        all_below = resample_fastlog_tags(**kwargs, require_all_busy_tags=False)
        assert any_below["busy_a"].isna().all()
        assert all_below["busy_a"].iloc[:-1].notna().all()


class TestResampleOptionsReachTheCacheLayer:
    """Resample options must be settable through the cached, day-chunked entry points.

    ``make_fl_resampled_one_device`` used to enumerate the options it forwarded, and that list
    went stale as ``resample_fastlog_tags`` gained more: ``circular_tags``, ``ffill_tags``,
    ``busy_tag_ffill_limit_s`` and ``require_all_busy_tags`` were all unreachable through the
    cache layer. ``circular_tags`` mattered most, since falling back to its hardcoded default
    silently averages a differently-named direction tag across the 0/360 wrap.
    """

    @pytest.fixture
    def spy_resample(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Capture the kwargs resample_fastlog_tags is called with, and stub out the raw load."""
        seen: dict = {}

        def fake_resample(**kwargs: object) -> pd.DataFrame:
            seen.update(kwargs)
            return pd.DataFrame(
                {"ActPower_Value": [1.0]}, index=pd.DatetimeIndex([DAY], name=TIMESTAMP_NAME, freq="1s")
            )

        monkeypatch.setattr(flh, "resample_fastlog_tags", fake_resample)
        monkeypatch.setattr(
            flh,
            "_get_raw_df_dict",
            lambda **_: {
                "ActPower_Value": pd.DataFrame(
                    {"ActPower_Value": [1.0]}, index=pd.DatetimeIndex([DAY], name=TIMESTAMP_NAME)
                )
            },
        )
        return seen

    @pytest.mark.parametrize(
        ("option", "value"),
        [
            ("circular_tags", ("AcWindDr_Value",)),
            ("ffill_tags", ()),
            ("std_tags", ("ActPower_Value",)),
            ("min_raw_data_count", 3),
            ("busy_tag_ffill_limit_s", 45),
            ("require_all_busy_tags", True),
        ],
    )
    def test_option_reaches_resample_fastlog_tags(self, spy_resample: dict, option: str, value: object) -> None:
        flh.make_fl_resampled_one_device(
            park_id=PARK,
            device_id=DEVICE,
            start_dt=DAY,
            end_dt_excl=DAY_END,
            siemens_parks={PARK},
            **{option: value},  # type: ignore[arg-type]
        )
        assert spy_resample[option] == value

    def test_option_reaches_through_the_day_chunking_and_cache(
        self, spy_resample: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # tz-aware, because this entry point coerces its result index to UTC before trimming it.
        monkeypatch.setattr(flh, "_max_source_mtime", lambda **_: None)
        flh.get_fl_resampled_one_device(
            park_id=PARK,
            device_id=DEVICE,
            start_dt=pd.Timestamp(DAY, tz="UTC"),
            end_dt_excl=pd.Timestamp(DAY_END, tz="UTC"),
            cache_dir=tmp_path,
            siemens_parks={PARK},
            require_all_busy_tags=True,
            busy_tag_ffill_limit_s=45,
        )
        assert spy_resample["require_all_busy_tags"] is True
        assert spy_resample["busy_tag_ffill_limit_s"] == 45


class TestCacheKeyStability:
    """Adding resample options must not orphan days cached before those options existed.

    The key is built by introspection over the named arguments, so a defaulted option has to be
    left out of it: hashing every option would mean merely widening the signature invalidated
    every cached day, over parameters whose defaults reproduce the previous behaviour exactly.
    """

    LEGACY_PARAMS: ClassVar[dict[str, Any]] = {
        "park_id": "HOT",
        "device_id": "2304512",
        "start_dt": pd.Timestamp("2024-01-01"),
        "end_dt_excl": pd.Timestamp("2024-01-02"),
        "timebase_s": 1,
        "tags": None,
        "busy_tags": None,
        "minmax_tags": None,
        "min_data_count": None,
    }
    # Measured on the commit before std_tags/min_raw_data_count were added. A change here means
    # every existing user's cache silently stops being found.
    LEGACY_KEY = "IZfqhKlUtfd8GBnRSyEDIWIdDgeRGuVH"

    def test_legacy_default_key_is_unchanged(self) -> None:
        assert flh.create_consistent_hash(**self.LEGACY_PARAMS) == self.LEGACY_KEY

    def test_defaulted_options_are_omitted_from_the_key(self) -> None:
        assert flh._non_default_resample_kwargs({}) == {}  # noqa: SLF001
        # Explicitly passing a default is indistinguishable from not passing it, by design.
        assert flh._non_default_resample_kwargs({"std_tags": None, "require_all_busy_tags": False}) == {}  # noqa: SLF001

    def test_non_default_options_are_kept_in_the_key(self) -> None:
        kept = flh._non_default_resample_kwargs(  # noqa: SLF001
            {"require_all_busy_tags": True, "min_raw_data_count": 3, "std_tags": None}
        )
        assert kept == {"require_all_busy_tags": True, "min_raw_data_count": 3}

    def test_an_unknown_option_is_kept_rather_than_silently_dropped(self) -> None:
        # A typo must not be swallowed; resample_fastlog_tags will raise on it anyway.
        assert flh._non_default_resample_kwargs({"not_a_real_option": 1}) == {"not_a_real_option": 1}  # noqa: SLF001

    def test_a_non_default_option_changes_the_key(self) -> None:
        with_option = dict(self.LEGACY_PARAMS) | {"require_all_busy_tags": True}
        assert flh.create_consistent_hash(**with_option) != self.LEGACY_KEY


CHUNK_START = pd.Timestamp("2024-03-01")
CHUNK_END = pd.Timestamp("2024-03-04")
CHUNK_KW: dict[str, Any] = {
    "busy_tags": ("ActPower_Value", "AcWindSp_AcWindSp"),
    "require_all_busy_tags": True,
    "busy_tag_ffill_limit_s": 45,
    "minmax_tags": ("ActPower_Value",),
    "std_tags": ("ActPower_Value",),
}


def _synthetic_raw() -> dict[str, pd.DataFrame]:
    """Three days of 10s data holding both outage shapes the chunked path has to survive.

    A row-arrival gap stops every tag at once (day 2); a dead busy tag stops the vane while power
    keeps logging (day 3), which no row-arrival check can see. Neither sits on a chunk boundary,
    so a chunk that lost its lead-in would mask differently from an unchunked run rather than
    failing outright.
    """
    idx = pd.date_range(CHUNK_START, CHUNK_END, freq="10s", inclusive="left", name=TIMESTAMP_NAME)
    counter = np.arange(len(idx))
    power = pd.Series(np.sin(counter / 97) * 1000 + 1500, index=idx)
    wind = pd.Series(np.cos(counter / 53) * 3 + 8, index=idx)
    gap = (idx >= pd.Timestamp("2024-03-02 04:00")) & (idx < pd.Timestamp("2024-03-02 04:20"))
    dead = (idx >= pd.Timestamp("2024-03-03 09:00")) & (idx < pd.Timestamp("2024-03-03 09:30"))
    return {
        "ActPower_Value": power[~gap].to_frame("ActPower_Value"),
        "AcWindSp_AcWindSp": wind[~gap & ~dead].to_frame("AcWindSp_AcWindSp"),
    }


def _slice_loader(raw: dict[str, pd.DataFrame]) -> flh.RawLoader:
    """Return a RawLoader serving a prepared raw dict, the way a real source serves a date range."""

    def load_raw(start_dt: pd.Timestamp, end_dt_excl: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return {tag: df[(df.index >= start_dt) & (df.index < end_dt_excl)] for tag, df in raw.items()}

    return load_raw


def _counting_stub(calls: list[int]) -> "object":
    """Return a _resample_one_chunk stand-in that records that it was called."""

    def stub(**_: object) -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame()

    return stub


class TestGetResampledChunkedCached:
    """The chunk/cache layer driven by an injected raw source rather than the Siemens filestore."""

    def _chunked(self, raw: dict[str, pd.DataFrame], **kwargs: Any) -> pd.DataFrame:  # noqa: ANN401
        return flh.get_resampled_chunked_cached(
            load_raw=_slice_loader(raw),
            start_dt=CHUNK_START,
            end_dt_excl=CHUNK_END,
            timebase_s=60,
            cache_key_extra={"source": "synthetic"},
            **(CHUNK_KW | kwargs),
        )

    def test_chunked_matches_unchunked(self) -> None:
        # What the lead-in is for: resampling three days one day at a time must give the same
        # answer, masking included, as resampling them together.
        raw = _synthetic_raw()
        whole = resample_fastlog_tags(raw_df_dict=raw, timebase_s=60, **CHUNK_KW)
        expected = whole[(whole.index >= CHUNK_START) & (whole.index < CHUNK_END)].resample("60s").last()

        actual = self._chunked(raw)

        pd.testing.assert_frame_equal(actual, expected, check_freq=False)

    def test_both_outages_are_masked(self) -> None:
        # Keeps the test above from passing vacuously: with nothing masked, chunked and unchunked
        # would agree trivially and the lead-in would not be under test at all. It also pins the
        # two masks' different reach -- a row-arrival gap voids the window, a dead busy tag voids
        # only itself, since the tags still reporting are still reporting the truth.
        actual = self._chunked(_synthetic_raw())
        assert actual.loc["2024-03-02 04:05":"2024-03-02 04:15"].isna().all().all()
        dead = slice("2024-03-03 09:05", "2024-03-03 09:25")
        assert actual.loc[dead, "AcWindSp_AcWindSp"].isna().all()
        assert actual.loc[dead, "ActPower_Value"].notna().all()
        assert actual["ActPower_Value"].notna().sum() > 4000  # and the rest of the three days survives

    def test_cache_hit_on_second_call(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        raw = _synthetic_raw()
        first = self._chunked(raw, cache_dir=tmp_path)
        calls: list[int] = []
        monkeypatch.setattr(flh, "_resample_one_chunk", _counting_stub(calls))
        second = self._chunked(raw, cache_dir=tmp_path)
        assert calls == []
        pd.testing.assert_frame_equal(first, second, check_freq=False)

    def test_changed_resample_option_misses_the_cache(self, tmp_path: Path) -> None:
        raw = _synthetic_raw()
        self._chunked(raw, cache_dir=tmp_path)
        self._chunked(raw, cache_dir=tmp_path, min_raw_data_count=3)
        assert len(list(tmp_path.glob("*.parquet"))) == 2 * 3  # two settings x three days

    def test_cache_key_extra_separates_sources(self, tmp_path: Path) -> None:
        # Two sources with identical resample options must not read each other's chunks.
        raw = _synthetic_raw()
        self._chunked(raw, cache_dir=tmp_path)
        flh.get_resampled_chunked_cached(
            load_raw=_slice_loader(raw),
            start_dt=CHUNK_START,
            end_dt_excl=CHUNK_END,
            timebase_s=60,
            cache_key_extra={"source": "other"},
            cache_dir=tmp_path,
            **CHUNK_KW,
        )
        assert len(list(tmp_path.glob("*.parquet"))) == 2 * 3

    def test_cache_subdir_is_used(self, tmp_path: Path) -> None:
        self._chunked(_synthetic_raw(), cache_dir=tmp_path, cache_subdir=("resampled", "synthetic"))
        assert len(list((tmp_path / "resampled" / "synthetic").glob("*.parquet"))) == 3

    def test_cache_key_is_stable_across_processes(self, tmp_path: Path) -> None:
        """A key built by introspecting an injected callable would move every run.

        ``create_consistent_hash`` falls back to ``str(obj)`` for anything it does not recognise,
        and a function or ``functools.partial`` stringifies with its memory address. The failure
        mode is a silent permanent cache miss, never an error, so it has to be checked across real
        processes rather than within one.
        """
        script = tmp_path / "run_once.py"
        script.write_text(
            "from functools import partial\n"
            "from pathlib import Path\n"
            "import sys\n"
            "import pandas as pd\n"
            "from hot_open.fastlog_helpers import get_resampled_chunked_cached\n"
            "def load_raw(start_dt, end_dt_excl, unused=None):\n"
            "    idx = pd.date_range('2024-03-01', periods=60, freq='10s', name='timestamp')\n"
            "    df = pd.DataFrame({'ActPower_Value': range(60)}, index=idx, dtype=float)\n"
            "    return {'ActPower_Value': df[(df.index >= start_dt) & (df.index < end_dt_excl)]}\n"
            "get_resampled_chunked_cached(\n"
            "    load_raw=partial(load_raw, unused=1),\n"
            "    start_dt=pd.Timestamp('2024-03-01'),\n"
            "    end_dt_excl=pd.Timestamp('2024-03-02'),\n"
            "    timebase_s=60,\n"
            "    cache_key_extra={'source': 'synthetic'},\n"
            "    cache_dir=Path(sys.argv[1]),\n"
            ")\n",
            encoding="utf-8",
        )
        cache = tmp_path / "cache"
        for _ in range(2):
            subprocess.run([sys.executable, str(script), str(cache)], check=True)  # noqa: S603
        assert len(list(cache.glob("*.parquet"))) == 1

    def test_source_mtime_none_keeps_a_cached_chunk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # An immutable source (a parquet datapack) must keep its chunks however old they look.
        raw = _synthetic_raw()
        self._chunked(raw, cache_dir=tmp_path)
        for parquet in tmp_path.glob("*.parquet"):
            os.utime(parquet, (0.0, 0.0))
        calls: list[int] = []
        monkeypatch.setattr(flh, "_resample_one_chunk", _counting_stub(calls))
        self._chunked(raw, cache_dir=tmp_path)
        assert calls == []

    def test_newer_source_mtime_invalidates_a_cached_chunk(self, tmp_path: Path) -> None:
        raw = _synthetic_raw()
        self._chunked(raw, cache_dir=tmp_path)
        newest = max(x.stat().st_mtime for x in tmp_path.glob("*.parquet"))
        asked: list[int] = []

        def source_mtime_fn(start_dt: pd.Timestamp, end_dt_excl: pd.Timestamp) -> float:  # noqa: ARG001
            asked.append(1)
            return newest + 1000

        before = len(list(tmp_path.glob("*.parquet")))
        self._chunked(raw, cache_dir=tmp_path, source_mtime_fn=source_mtime_fn)
        assert len(asked) == 3  # every chunk checked
        assert len(list(tmp_path.glob("*.parquet"))) == before  # recomputed in place, same key

    def test_no_cache_dir_still_works(self) -> None:
        # Caching is optional; the chunking is not.
        assert not self._chunked(_synthetic_raw()).empty

    def test_empty_source_returns_an_empty_frame(self, tmp_path: Path) -> None:
        empty = flh.get_resampled_chunked_cached(
            load_raw=lambda *_: {},
            start_dt=CHUNK_START,
            end_dt_excl=CHUNK_END,
            timebase_s=60,
            cache_key_extra={"source": "synthetic"},
            cache_dir=tmp_path,
        )
        assert empty.empty
        assert list(tmp_path.glob("*.parquet")) == []  # nothing cached, so a later real load recomputes


class TestSiemensWrapperOverGenericLayer:
    """``get_fl_resampled_one_device`` is a caller of the generic layer, not its own copy of it."""

    @pytest.fixture
    def stub_chunk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stand in for the resample, so these tests are about the wrapper and nothing else."""
        idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01")], name=TIMESTAMP_NAME)
        monkeypatch.setattr(flh, "_resample_one_chunk", lambda **_: pd.DataFrame({"ActPower_Value": [1.0]}, index=idx))
        monkeypatch.setattr(flh, "_max_source_mtime", lambda **_: None)

    @pytest.mark.usefixtures("stub_chunk")
    def test_end_to_end_cache_key_is_unchanged(self, tmp_path: Path) -> None:
        # The strongest form of the pin below: not just that the hash function is stable, but that
        # the wrapper still feeds it the same parameter set. A change orphans every cached day.
        get_fl_resampled_one_device(
            park_id="HOT",
            device_id="2304512",
            start_dt=pd.Timestamp("2024-01-01", tz="UTC"),
            end_dt_excl=pd.Timestamp("2024-01-02", tz="UTC"),
            timebase_s=1,
            cache_dir=tmp_path,
        )
        cached = list((tmp_path / "fl_resampled" / "HOT" / "2304512").glob("*.parquet"))
        assert [x.name for x in cached] == [f"20240101_{TestCacheKeyStability.LEGACY_KEY}.parquet"]

    @pytest.mark.usefixtures("stub_chunk")
    def test_naive_chunks_are_localised_for_a_tz_aware_range(self, tmp_path: Path) -> None:
        # Chunks are cut on naive calendar dates and HOT fastlog files carry naive timestamps, but
        # every HOT caller asks in UTC -- so the result has to come back tz-aware.
        result = get_fl_resampled_one_device(
            park_id="HOT",
            device_id="2304512",
            start_dt=pd.Timestamp("2024-01-01", tz="UTC"),
            end_dt_excl=pd.Timestamp("2024-01-02", tz="UTC"),
            cache_dir=tmp_path,
        )
        assert str(result.index.tz) == "UTC"  # type: ignore[attr-defined]
