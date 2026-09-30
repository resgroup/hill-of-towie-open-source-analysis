"""Tests for writing a parquet file without ever exposing a half-written one."""

import os
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pyarrow.lib import ArrowInvalid

from hot_open import parquet_io
from hot_open.parquet_io import write_parquet_atomic


@pytest.fixture
def df() -> pd.DataFrame:
    """Return a small frame with the DatetimeIndex a cached chunk carries."""
    return pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.date_range("2024-03-01", periods=3, freq="10min"))


class TestWriteParquetAtomic:
    """A cache keyed on a file's existence needs that file to appear complete or not at all."""

    def test_round_trips(self, tmp_path: Path, df: pd.DataFrame) -> None:
        path = tmp_path / "chunk.parquet"

        write_parquet_atomic(df, path)

        # check_freq: a parquet round trip does not carry a DatetimeIndex's freq attribute.
        pd.testing.assert_frame_equal(pd.read_parquet(path), df, check_freq=False)

    def test_creates_the_parent_directory(self, tmp_path: Path, df: pd.DataFrame) -> None:
        path = tmp_path / "fl_resampled" / "PARK" / "WTG01" / "chunk.parquet"

        write_parquet_atomic(df, path)

        assert path.is_file()

    def test_the_final_path_is_untouched_until_the_write_completes(
        self, tmp_path: Path, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole point: a reader checking existence must not catch a file mid-write.

        A parquet's footer and magic bytes are written last, so writing straight to the final
        path publishes an unreadable file for the duration of the write.
        """
        path = tmp_path / "chunk.parquet"
        seen: list[bool] = []
        real_to_parquet = pd.DataFrame.to_parquet

        def spy(frame: pd.DataFrame, target: Path, **kwargs: object) -> None:
            seen.append(path.exists())  # is the final path published while we are still writing?
            real_to_parquet(frame, target, **kwargs)  # type: ignore[call-overload]

        monkeypatch.setattr(pd.DataFrame, "to_parquet", spy)
        write_parquet_atomic(df, path)

        assert seen == [False]
        assert path.is_file()

    def test_a_failed_write_leaves_nothing_behind(
        self, tmp_path: Path, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A writer killed part-way must not leave a file for later runs to serve, nor a temp."""
        path = tmp_path / "chunk.parquet"

        def fail(_self: pd.DataFrame, target: Path, **_kwargs: object) -> None:
            Path(target).write_bytes(b"PAR1 truncated")  # a plausible-looking partial file
            msg = "disk full"
            raise ArrowInvalid(msg)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
        with pytest.raises(ArrowInvalid):
            write_parquet_atomic(df, path)

        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_a_failed_rewrite_leaves_the_previous_file_readable(
        self, tmp_path: Path, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache-poisoning case: a failed refresh must not destroy a good cached chunk."""
        path = tmp_path / "chunk.parquet"
        write_parquet_atomic(df, path)

        def fail(_self: pd.DataFrame, target: Path, **_kwargs: object) -> None:
            Path(target).write_bytes(b"PAR1 truncated")
            msg = "interrupted"
            raise ArrowInvalid(msg)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
        with pytest.raises(ArrowInvalid):
            write_parquet_atomic(df, path)

        pd.testing.assert_frame_equal(pd.read_parquet(path), df, check_freq=False)

    def test_two_writers_of_the_same_path_do_not_collide(self, tmp_path: Path, df: pd.DataFrame) -> None:
        """Temp names are per-process, so parallel runs over one cache key stay valid."""
        path = tmp_path / "chunk.parquet"

        write_parquet_atomic(df, path)
        write_parquet_atomic(df * 2, path)

        pd.testing.assert_frame_equal(pd.read_parquet(path), df * 2, check_freq=False)
        assert list(tmp_path.iterdir()) == [path]

    def test_concurrent_threads_writing_one_path_each_get_their_own_temp(
        self, tmp_path: Path, df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Threads share a pid, so the temp name needs more than the pid to stay unique.

        Both writers are held inside ``to_parquet`` at once, which is the window that matters: with
        a pid-only temp name they write the same file, interleaving into a corrupt parquet, and the
        first to finish deletes the second's temp in its ``finally``.
        """
        path = tmp_path / "chunk.parquet"
        both_inside = threading.Barrier(2, timeout=10)
        real_to_parquet = pd.DataFrame.to_parquet
        temp_names: list[str] = []

        def slow(self: pd.DataFrame, where: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            temp_names.append(Path(where).name)
            both_inside.wait()  # neither returns until the other has also claimed its temp path
            return real_to_parquet(self, where, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", slow)
        errors: list[BaseException] = []

        def write(frame: pd.DataFrame) -> None:
            try:
                write_parquet_atomic(frame, path)
            except BaseException as e:  # noqa: BLE001 -- re-raised in the main thread below
                errors.append(e)

        threads = [threading.Thread(target=write, args=(x,)) for x in (df, df * 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        assert len(set(temp_names)) == 2, f"threads shared a temp path: {temp_names}"
        # Whichever renamed last wins, but it must be one complete frame and no temp left behind.
        written = pd.read_parquet(path)
        assert any(written.equals(x) for x in (df, df * 2))
        assert list(tmp_path.iterdir()) == [path]


class TestReplaceWithRetry:
    """Windows' rename is atomic but not concurrent, so contention has to be waited out.

    The threaded test above exercises this for real but only when the timing lands; these pin the
    behaviour deterministically.
    """

    def test_a_contended_replace_is_retried(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        src, dst = tmp_path / "a.tmp", tmp_path / "a"
        src.write_text("payload")
        calls = []
        real_replace = os.replace

        def flaky(a: Any, b: Any) -> None:  # noqa: ANN401
            calls.append(a)
            if len(calls) < 3:
                msg = "Access is denied"
                raise PermissionError(13, msg)
            real_replace(a, b)

        monkeypatch.setattr(os, "replace", flaky)
        monkeypatch.setattr(parquet_io, "_REPLACE_BACKOFF_S", 0)

        parquet_io._replace_with_retry(src, dst)  # noqa: SLF001

        assert len(calls) == 3
        assert dst.read_text() == "payload"

    def test_a_permission_error_that_never_clears_is_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A locked file or a real permissions problem is not contention; it must not be swallowed."""
        src, dst = tmp_path / "a.tmp", tmp_path / "a"
        src.write_text("payload")

        def always_denied(*_: Any) -> None:  # noqa: ANN401
            msg = "Access is denied"
            raise PermissionError(13, msg)

        monkeypatch.setattr(os, "replace", always_denied)
        monkeypatch.setattr(parquet_io, "_REPLACE_BACKOFF_S", 0)

        with pytest.raises(PermissionError):
            parquet_io._replace_with_retry(src, dst)  # noqa: SLF001
