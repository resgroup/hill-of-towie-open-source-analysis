"""Helpers to download data from external sources."""

import json
import logging
from collections.abc import Collection
from pathlib import Path

import requests
from tqdm import tqdm

from hot_open.paths import DATA_DIR

logger = logging.getLogger(__name__)

BYTES_IN_1MB = 1024 * 1024
CHUNK_SIZE = 10 * BYTES_IN_1MB


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
        with metadata_fpath.open() as f:
            content = json.load(f)
    else:
        logger.info("Fetching metadata from zenodo...")
        r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=10)
        r.raise_for_status()
        content = r.json()
        with metadata_fpath.open("w") as f:
            json.dump(content, f)

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
                    total=round(_file_size / CHUNK_SIZE, 3),
                    unit="MB",
                    unit_scale=10,  # 10 MB per iteration
                    desc=f"Downloading {_file_name} ({_file_size / BYTES_IN_1MB:.2f} MB)",
                ):
                    f.write(chunk)
            downloaded_files += 1
        else:
            logger.info("[%d/%d] File %s already exists. Skipping download.", i_file, n_files_to_download, dst_fpath)

    logger.info("Download finished: %s new files cached at %s", downloaded_files, output_dir)


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
