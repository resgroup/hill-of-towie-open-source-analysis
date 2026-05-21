"""Helpers to download data from external sources."""

import json
import logging
import math
import shutil
import time
import zipfile
from collections.abc import Collection
from pathlib import Path
from typing import NamedTuple

import requests
from tqdm import tqdm

from hot_open.settings import get_data_dir

logger = logging.getLogger(__name__)

BYTES_IN_1MB = 1024 * 1024
CHUNK_SIZE = 10 * BYTES_IN_1MB
SMALL_FILE_THRESHOLD_BYTES = 2 * BYTES_IN_1MB
_HOT_V2_RECORD_ID = "20204946"

# Network resilience knobs for streamed Zenodo downloads.
# ``timeout`` is passed to ``requests.get`` as a ``(connect, read)`` tuple.
# With ``stream=True`` the read value is the budget *between* bytes received
# from the server; 60 s tolerates short Zenodo stalls without hanging an
# overnight run on a truly dead connection.
_CONNECT_TIMEOUT_S = 10
_READ_TIMEOUT_S = 60
_MAX_DOWNLOAD_ATTEMPTS = 5
_BACKOFF_BASE_S = 2.0
# Exceptions worth retrying. These are transient network errors that may
# resolve on a new TCP connection. Non-retryable cases (4xx/5xx HTTP errors
# from ``raise_for_status``) fall through to the required-vs-optional
# branching below.
_RETRYABLE_EXCEPTIONS: tuple[type[requests.RequestException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
)
_HTTP_PARTIAL_CONTENT = 206


def download_zenodo_data(
    record_id: str,
    output_dir: Path | None = None,
    filenames: Collection[str] | None = None,
    *,
    cache_overwrite: bool = False,
) -> None:
    """Download and caches files from zenodo.org."""
    output_dir = output_dir if output_dir is not None else get_data_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_fpath = output_dir / "zenodo_dataset_metadata.json"

    if not cache_overwrite and metadata_fpath.is_file():
        msg = f"Loading metadata from {metadata_fpath}"
        logger.info(msg)
        with metadata_fpath.open() as f:
            content = json.load(f)
    else:
        logger.info("Fetching metadata from zenodo...")
        r = requests.get(
            f"https://zenodo.org/api/records/{record_id}",
            timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
        )
        r.raise_for_status()
        content = r.json()
        with metadata_fpath.open("w") as f:
            json.dump(content, f)
        msg = f"Saved metadata to {metadata_fpath}"
        logger.info(msg)

    remote_files: list[dict] = content["files"]
    if filenames is None:
        files_to_download: list[dict] = list(remote_files)
    else:
        files_to_download = list(_check_name_of_files_to_download(filenames, remote_files))
    required_keys = {f["key"] for f in files_to_download}
    # Auto-include any small file in the record so users can explore READMEs,
    # deployment reports, etc. without an extra fetch step.
    for rf in remote_files:
        if rf["size"] < SMALL_FILE_THRESHOLD_BYTES and rf["key"] not in required_keys:
            files_to_download.append(rf)

    downloaded_files = 0
    n_files_to_download = len(files_to_download)

    for i_file, file_to_download in enumerate(files_to_download, start=1):
        is_required = file_to_download["key"] in required_keys
        downloaded_files += _download_one_file(
            file_to_download,
            output_dir,
            cache_overwrite=cache_overwrite,
            is_required=is_required,
            progress_prefix=f"[{i_file}/{n_files_to_download}]",
        )

    logger.info("Download finished: %s new files cached at %s", downloaded_files, output_dir)


def _download_one_file(
    file_entry: dict,
    output_dir: Path,
    *,
    cache_overwrite: bool,
    is_required: bool,
    progress_prefix: str,
) -> int:
    """Download a single Zenodo file. Returns 1 if a new file was written, 0 otherwise.

    Retries up to ``_MAX_DOWNLOAD_ATTEMPTS`` times on transient network errors
    (connection drops, read timeouts, chunked-encoding errors), with exponential
    backoff between attempts. Each retry sends ``Range: bytes=<existing>-`` so a
    partially downloaded file resumes from where the previous attempt left off
    rather than restarting from byte 0.

    After exhausting attempts: required files re-raise the last
    ``requests.RequestException`` (with any partial bytes left on disk so a
    subsequent run can resume); optional files log a warning, remove partial
    bytes, and return 0.
    """
    _file_name = file_entry["key"]
    _file_size = file_entry["size"]
    _file_url = file_entry["links"]["self"]
    dst_fpath = output_dir / _file_name

    if cache_overwrite and dst_fpath.is_file():
        dst_fpath.unlink()

    already_complete = dst_fpath.is_file() and dst_fpath.stat().st_size >= _file_size
    if already_complete:
        logger.info("%s File %s already exists. Skipping download.", progress_prefix, dst_fpath)
        return 0

    logger.info("%s Beginning file download from Zenodo: %s...", progress_prefix, _file_name)
    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
        is_last_attempt = attempt == _MAX_DOWNLOAD_ATTEMPTS
        existing_size = dst_fpath.stat().st_size if dst_fpath.is_file() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
        try:
            result = requests.get(
                _file_url,
                stream=True,
                timeout=(_CONNECT_TIMEOUT_S, _READ_TIMEOUT_S),
                headers=headers,
            )
            result.raise_for_status()
            # If we requested a Range but the server returned 200 instead of 206,
            # it ignored the Range header — restart from byte 0.
            resume = existing_size > 0 and result.status_code == _HTTP_PARTIAL_CONTENT
            if existing_size > 0 and not resume:
                logger.info(
                    "%s Server did not honor Range request (status %s); restarting from byte 0.",
                    progress_prefix,
                    result.status_code,
                )
                existing_size = 0
            file_mode = "ab" if resume else "wb"
            remaining_bytes = max(0, _file_size - existing_size)
            with (
                Path.open(dst_fpath, file_mode) as f,
                tqdm(
                    total=remaining_bytes,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {_file_name} ({_file_size / BYTES_IN_1MB:.2f} MB)",
                ) as pbar,
            ):
                for chunk in result.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))
        except _RETRYABLE_EXCEPTIONS as e:
            if not is_last_attempt:
                partial_size = dst_fpath.stat().st_size if dst_fpath.is_file() else 0
                sleep_s = _BACKOFF_BASE_S * (2 ** (attempt - 1))
                logger.warning(
                    "%s Download attempt %d/%d for %s failed (%s). Have %d/%d bytes. Sleeping %.1fs before retrying.",
                    progress_prefix,
                    attempt,
                    _MAX_DOWNLOAD_ATTEMPTS,
                    _file_name,
                    e,
                    partial_size,
                    _file_size,
                    sleep_s,
                )
                time.sleep(sleep_s)
                continue
            # Exhausted retries on a transient error — fall through to the
            # required-vs-optional resolution below.
            return _resolve_download_failure(
                exc=e,
                dst_fpath=dst_fpath,
                file_name=_file_name,
                is_required=is_required,
                progress_prefix=progress_prefix,
            )
        except requests.RequestException as e:
            # Non-retryable (e.g. 4xx HTTPError). Resolve immediately.
            return _resolve_download_failure(
                exc=e,
                dst_fpath=dst_fpath,
                file_name=_file_name,
                is_required=is_required,
                progress_prefix=progress_prefix,
            )
        else:
            return 1

    msg = "unreachable: retry loop should have returned or raised"
    raise RuntimeError(msg)


def _resolve_download_failure(
    *,
    exc: requests.RequestException,
    dst_fpath: Path,
    file_name: str,
    is_required: bool,
    progress_prefix: str,
) -> int:
    """Resolve a download failure: re-raise for required files, warn-and-clean for optional ones."""
    if is_required:
        # Leave partial bytes on disk so a subsequent run can resume via Range.
        raise exc
    logger.warning(
        "%s Failed to download optional small file %s: %s. Continuing.",
        progress_prefix,
        file_name,
        exc,
    )
    if dst_fpath.is_file():
        dst_fpath.unlink()
    return 0


def _missing_small_files_from_cached_metadata(target_dir: Path) -> list[str]:
    """Return small-file keys absent from ``target_dir`` based on cached Zenodo metadata.

    Returns an empty list when the metadata cache is missing or unreadable; the
    next successful Zenodo fetch rewrites the cache.
    """
    metadata_fpath = target_dir / "zenodo_dataset_metadata.json"
    if not metadata_fpath.is_file():
        return []
    try:
        with metadata_fpath.open() as f:
            content = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return [
        rf["key"]
        for rf in content.get("files", [])
        if rf.get("size", math.inf) < SMALL_FILE_THRESHOLD_BYTES and not (target_dir / rf["key"]).is_file()
    ]


def ensure_hot_data_files(
    filenames: Collection[str],
    *,
    data_dir: Path | None = None,
) -> None:
    """Download missing Hill of Towie v2 data files from Zenodo.

    Idempotent: files already present in ``data_dir`` are not re-downloaded
    and no network call is made when every requested file exists locally and
    the cached metadata shows no missing small files.

    Auto-includes every file in the record under 2 MB so users can browse
    READMEs, deployment reports, etc. Failures on these auto-included
    downloads warn-and-continue (handled inside ``download_zenodo_data``).
    """
    target_dir = data_dir if data_dir is not None else get_data_dir()
    requested = list(filenames)
    missing_requested = [f for f in requested if not (target_dir / f).is_file()]
    missing_small = _missing_small_files_from_cached_metadata(target_dir)
    if not missing_requested and not missing_small:
        logger.info(
            "ensure_hot_data_files: all %d requested files already present at %s "
            "(and no missing small files in cached metadata), skipping download",
            len(requested),
            target_dir,
        )
        return
    logger.info(
        "ensure_hot_data_files: downloading from Zenodo record %s into %s "
        "(missing requested: %s; missing small files via cached metadata: %s)",
        _HOT_V2_RECORD_ID,
        target_dir,
        missing_requested,
        missing_small,
    )
    download_zenodo_data(record_id=_HOT_V2_RECORD_ID, output_dir=target_dir, filenames=missing_requested)


class _ZipLayout(NamedTuple):
    """Per-zip extraction layout.

    ``top_level`` is the directory created by the zip's top-level entries; it is
    returned to the caller as the "where the data lives" path.

    ``sentinel`` is a deeper sub-path (relative to ``data_dir``) used for the
    idempotency check. It must be a path that ONLY successful extraction
    creates — never something a stray ``mkdir`` elsewhere in the codebase
    can produce. In particular ``get_filestore_dir()`` eagerly creates
    ``turbine_fastlog/Filestore/`` via ``mkdir(parents=True, exist_ok=True)``,
    so the fastlog sentinel must be deeper than that.
    """

    top_level: str
    sentinel: str


# Map of known Zenodo data zips to their extraction layout. When extending,
# also ensure the file is published in the v2 record.
_ZIP_LAYOUT: dict[str, _ZipLayout] = {
    "turbine_fastlog.zip": _ZipLayout(
        top_level="turbine_fastlog",
        sentinel="turbine_fastlog/Filestore/FL",
    ),
    "lidar_data.zip": _ZipLayout(
        top_level="lidar_data",
        sentinel="lidar_data/timeseries",
    ),
}


def ensure_extracted(zip_name: str, *, data_dir: Path | None = None) -> Path:
    """Download (if needed) and extract a Hill of Towie data zip under ``data_dir``.

    Returns the path to the extracted top-level directory.

    Idempotent: if the extracted directory already exists, returns it immediately
    with no download or extraction. The source zip is deleted after a successful
    extraction. On extraction failure, any partial top-level directory is removed
    before the exception is re-raised so a subsequent call re-attempts cleanly.
    """
    if zip_name not in _ZIP_LAYOUT:
        msg = f"Unknown Hill of Towie zip: {zip_name!r}. Known: {sorted(_ZIP_LAYOUT)}"
        raise ValueError(msg)
    target_dir = data_dir if data_dir is not None else get_data_dir()
    layout = _ZIP_LAYOUT[zip_name]
    top_level = target_dir / layout.top_level
    sentinel = target_dir / layout.sentinel
    if sentinel.exists():
        logger.info(
            "ensure_extracted: %s already extracted (sentinel %s present), skipping",
            zip_name,
            sentinel,
        )
        return top_level

    logger.info(
        "ensure_extracted: %s not extracted yet (sentinel %s missing), will download and extract under %s",
        zip_name,
        sentinel,
        target_dir,
    )
    ensure_hot_data_files([zip_name], data_dir=target_dir)
    zip_path = target_dir / zip_name
    logger.info("ensure_extracted: extracting %s into %s", zip_path, target_dir)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)
    except Exception:
        if top_level.exists():
            shutil.rmtree(top_level)
        raise
    zip_path.unlink()
    logger.info("ensure_extracted: extraction complete, deleted source zip %s", zip_path)
    return top_level


def _check_name_of_files_to_download(filenames: Collection[str], remote_files: Collection[dict]) -> Collection[dict]:
    requested_filenames = set(filenames)
    remote_filenames = {i["key"] for i in remote_files}
    if not requested_filenames.issubset(remote_filenames):
        msg = (
            "Could not find all files in the Zenodo record. "
            f"Missing files: {requested_filenames.difference(remote_filenames)}"
        )
        raise ValueError(msg)
    return [i for i in remote_files if i["key"] in requested_filenames]
