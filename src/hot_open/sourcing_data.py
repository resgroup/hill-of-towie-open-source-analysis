"""Helpers to download data from external sources."""

import json
import logging
import math
import shutil
import zipfile
from collections.abc import Collection
from pathlib import Path

import requests
from tqdm import tqdm

from hot_open.paths import DATA_DIR

logger = logging.getLogger(__name__)

BYTES_IN_1MB = 1024 * 1024
CHUNK_SIZE = 10 * BYTES_IN_1MB
_HOT_V2_RECORD_ID = "20204946"


def download_zenodo_data(
    record_id: str,
    output_dir: Path = DATA_DIR,
    filenames: Collection[str] | None = None,
    *,
    cache_overwrite: bool = False,
) -> None:
    """Download and caches files from zenodo.org."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_fpath = output_dir / "zenodo_dataset_metadata.json"

    if not cache_overwrite and metadata_fpath.is_file():
        msg = f"Loading metadata from {metadata_fpath}"
        logger.info(msg)
        with metadata_fpath.open() as f:
            content = json.load(f)
    else:
        logger.info("Fetching metadata from zenodo...")
        r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=10)
        r.raise_for_status()
        content = r.json()
        with metadata_fpath.open("w") as f:
            json.dump(content, f)
        msg = f"Saved metadata to {metadata_fpath}"
        logger.info(msg)

    remote_files: list[dict] = content["files"]
    files_to_download = remote_files if not filenames else _check_name_of_files_to_download(filenames, remote_files)

    downloaded_files = 0
    n_files_to_download = len(files_to_download)

    for i_file, file_to_download in enumerate(files_to_download, start=1):
        _file_name = file_to_download["key"]
        _file_size = file_to_download["size"]
        _file_url = file_to_download["links"]["self"]

        dst_fpath = output_dir / _file_name
        if cache_overwrite or not dst_fpath.is_file() or dst_fpath.stat().st_size < _file_size:
            logger.info("[%d/%d] Beginning file download from Zenodo: %s...", i_file, n_files_to_download, _file_name)

            result = requests.get(_file_url, stream=True, timeout=10)
            with Path.open(dst_fpath, "wb") as f:
                for chunk in tqdm(
                    result.iter_content(chunk_size=CHUNK_SIZE),
                    total=math.ceil(_file_size / CHUNK_SIZE),
                    unit="MB",
                    unit_scale=10,  # 10 MB per iteration
                    desc=f"Downloading {_file_name} ({_file_size / BYTES_IN_1MB:.2f} MB)",
                ):
                    f.write(chunk)
            downloaded_files += 1
        else:
            logger.info("[%d/%d] File %s already exists. Skipping download.", i_file, n_files_to_download, dst_fpath)

    logger.info("Download finished: %s new files cached at %s", downloaded_files, output_dir)


def ensure_hot_data_files(
    filenames: Collection[str],
    *,
    data_dir: Path | None = None,
) -> None:
    """Download missing Hill of Towie v2 data files from Zenodo.

    Idempotent: files already present in ``data_dir`` are not re-downloaded
    and no network call is made when every requested file exists locally.
    """
    target_dir = data_dir if data_dir is not None else DATA_DIR
    requested = list(filenames)
    missing = [f for f in requested if not (target_dir / f).is_file()]
    if not missing:
        logger.info(
            "ensure_hot_data_files: all %d requested files already present at %s, skipping download",
            len(requested),
            target_dir,
        )
        return
    logger.info(
        "ensure_hot_data_files: %d of %d requested files missing at %s, downloading from Zenodo record %s: %s",
        len(missing),
        len(requested),
        target_dir,
        _HOT_V2_RECORD_ID,
        missing,
    )
    download_zenodo_data(record_id=_HOT_V2_RECORD_ID, output_dir=target_dir, filenames=missing)


# Map of known Zenodo data zips to the top-level directory created on extraction.
# This top-level directory also serves as the idempotency sentinel: if it exists,
# extraction is skipped. Callers pass the parent directory where this top-level
# directory should live as ``data_dir``.
# When extending, also ensure the file is published in the v2 record.
_ZIP_TOP_LEVEL_DIR: dict[str, str] = {
    "turbine_fastlog.zip": "turbine_fastlog",
    "lidar_data.zip": "lidar_data",
}


def ensure_extracted(zip_name: str, *, data_dir: Path | None = None) -> Path:
    """Download (if needed) and extract a Hill of Towie data zip under ``data_dir``.

    Returns the path to the extracted top-level directory.

    Idempotent: if the extracted directory already exists, returns it immediately
    with no download or extraction. The source zip is deleted after a successful
    extraction. On extraction failure, any partial top-level directory is removed
    before the exception is re-raised so a subsequent call re-attempts cleanly.
    """
    if zip_name not in _ZIP_TOP_LEVEL_DIR:
        msg = f"Unknown Hill of Towie zip: {zip_name!r}. Known: {sorted(_ZIP_TOP_LEVEL_DIR)}"
        raise ValueError(msg)
    target_dir = data_dir if data_dir is not None else DATA_DIR
    top_level = target_dir / _ZIP_TOP_LEVEL_DIR[zip_name]
    if top_level.exists():
        logger.info(
            "ensure_extracted: %s already exists at %s, skipping download and extraction",
            zip_name,
            top_level,
        )
        return top_level

    logger.info(
        "ensure_extracted: %s not found at %s, will download and extract under %s",
        zip_name,
        top_level,
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
