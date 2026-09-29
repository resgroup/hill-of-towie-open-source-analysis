"""Writing parquet caches so a reader never meets a half-written file."""

import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def write_parquet_atomic(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:  # noqa: ANN401
    """Write ``df`` to ``path`` via a temporary file beside it, then rename it into place.

    A parquet's footer and magic bytes are written last, so writing straight to the final path
    publishes a file that exists, is named correctly, and cannot be read, for as long as the write
    takes. Both of this module's callers key their caches on that name existing, which turns the
    window into two real failures:

    - a concurrent reader gets ``ArrowInvalid: Parquet magic bytes not found in footer``;
    - a writer killed part-way (interrupt, OOM, machine sleep) leaves the partial file behind for
      good. Nothing later re-checks it -- existence is the whole test, and its mtime is newer than
      the source it was built from -- so every subsequent run serves the corrupt chunk until
      somebody deletes it by hand.

    ``os.replace`` is atomic within a filesystem on POSIX and Windows alike, so a reader sees
    either the previous file or the complete new one. The temporary name is unique per call, so
    concurrent writers of one cache key cannot tread on each other's partial file; the last rename
    wins and both files were complete.

    Extra keyword arguments are passed through to :meth:`pandas.DataFrame.to_parquet`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        df.to_parquet(tmp_path, **kwargs)
        _replace_with_retry(tmp_path, path)
    finally:
        # A successful replace leaves nothing to remove; a failed write must not leave a temp.
        tmp_path.unlink(missing_ok=True)


_REPLACE_ATTEMPTS = 10
_REPLACE_BACKOFF_S = 0.05


def _replace_with_retry(tmp_path: Path, path: Path) -> None:
    """``os.replace``, retried briefly around Windows' non-exclusive rename.

    On Windows a rename onto a destination another writer is replacing, or a reader has open,
    raises ``PermissionError``. Every contender carries a complete file, so retrying is safe. A
    ``PermissionError`` that outlasts the attempts is a lock or a permissions problem and is raised.
    """
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
