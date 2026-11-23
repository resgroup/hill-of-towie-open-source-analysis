"""Reading and writing local parquet files safely."""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.fs

logger = logging.getLogger(__name__)


def read_parquet(path: Path | str, **kwargs: Any) -> pd.DataFrame:  # noqa: ANN401
    """Read a local parquet file into a DataFrame; use this instead of :func:`pandas.read_parquet`.

    pandas opens a local path as a Python file handle, and pyarrow can then release that handle's
    buffers on a worker thread while the interpreter exits, deadlocking the process on Python 3.13+.
    Reading through pyarrow's own local file system avoids it. Extra keyword arguments are passed to
    :func:`pandas.read_parquet`.
    """
    return pd.read_parquet(str(path), filesystem=pyarrow.fs.LocalFileSystem(), **kwargs)


_REPLACE_ATTEMPTS = 10
_REPLACE_BACKOFF_S = 0.05


def write_parquet_atomic(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:  # noqa: ANN401
    """Write ``df`` to ``path`` via a uniquely named temporary file beside it, then rename it into place.

    A reader sees either the previous file or the complete new one, and an interrupted write leaves
    no partial file for a cache's existence check to accept. Extra keyword arguments are passed to
    :meth:`pandas.DataFrame.to_parquet`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        df.to_parquet(tmp_path, **kwargs)
        _replace_with_retry(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _replace_with_retry(tmp_path: Path, path: Path) -> None:
    """``os.replace``, retried briefly: on Windows a rename onto a file another process has open raises."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp_path, path)  # noqa: PTH105 -- Path has no atomic-replace equivalent
        except PermissionError:  # noqa: PERF203 -- a retry loop is what this is
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            logger.debug("Replace of %s contended, retrying (attempt %d)", path, attempt + 1)
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
        else:
            return
