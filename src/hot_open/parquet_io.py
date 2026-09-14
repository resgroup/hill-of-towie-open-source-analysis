"""Writing parquet caches so a reader never meets a half-written file."""

import logging
import os
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
    either the previous file or the complete new one. The temporary name carries the process id,
    so two processes writing the same cache key cannot tread on each other's partial file; the
    last rename wins and both files were complete.

    Extra keyword arguments are passed through to :meth:`pandas.DataFrame.to_parquet`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        df.to_parquet(tmp_path, **kwargs)
        os.replace(tmp_path, path)  # noqa: PTH105 -- Path has no atomic-replace equivalent
    finally:
        # A successful replace leaves nothing to remove; a failed write must not leave a temp.
        tmp_path.unlink(missing_ok=True)
