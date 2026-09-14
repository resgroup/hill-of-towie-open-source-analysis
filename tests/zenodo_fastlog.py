"""Materialise fastlog fixture days from the Zenodo record, without downloading 11.9 GB.

The fastlog archive on the Hill of Towie record is ``turbine_fastlog.zip``, 11.9 GB. A test that
needs two turbine-days of it must not commit them -- that is several MB of binary in the repo
forever, for data that is already published -- but it must not download the whole archive either:
that is larger than a GitHub Actions cache (10 GB per repository) and far larger than a runner
wants to hold.

Zenodo answers HTTP range requests (``206``) on file content, so the middle path works: read the
zip's central directory over a range request, then fetch only the members wanted. Reading the
directory of the 11.9 GB archive costs about 5 s and 31,329 entries; the members these tests want
total 8.6 MB. ``remotezip`` does the range plumbing, and the archive is Zip64, which it handles.

Fetched files land under the data directory, not the repo, laid out exactly as
``ensure_extracted("turbine_fastlog.zip")`` would leave them (``FL/HOT/<device>/<day>/``) so
``get_fl_resampled(filestore_dir=...)`` reads them with no special case. The path carries the
record id, so publishing a new datapack misses rather than silently serving the old data.

CI caches the data directory; see ``.github/workflows/ci.yaml``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from remotezip import RemoteZip

from hot_open.settings import get_data_dir
from hot_open.sourcing_data import _HOT_V2_RECORD_ID

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)

FASTLOG_ZIP = "turbine_fastlog.zip"
ZENODO_CONTENT_URL = "https://zenodo.org/api/records/{record_id}/files/{filename}/content"

# Inside the archive every fastlog file sits under this prefix; the tests want the tree below it.
_ZIP_FILESTORE_PREFIX = "turbine_fastlog/Filestore/"


def fastlog_fixture_dir(data_dir: Path | None = None) -> Path:
    """Where fetched fixture days live: a filestore root holding ``FL/HOT/<device>/<day>/``.

    Keyed by record id so a new datapack does not silently reuse the previous one's files.
    """
    root = data_dir if data_dir is not None else get_data_dir()
    return root / f"fl_fixtures_{_HOT_V2_RECORD_ID}"


def ensure_fastlog_days(
    *,
    device_id: str,
    days: Mapping[str, Sequence[str]],
    data_dir: Path | None = None,
) -> Path:
    """Ensure the given tags of the given days are present locally, and return the filestore root.

    :param device_id: turbine serial, e.g. ``"2304510"``
    :param days: ``"YYYY-MM-DD"`` to the fastlog tags wanted that day, e.g. ``["YawPos_Value"]``
    :param data_dir: where to materialise them; defaults to the hot_open data directory
    :return: a directory containing ``FL/HOT/<device>/<day>/``, for ``filestore_dir=``

    Idempotent and offline once populated: the network is touched only when a wanted file is
    missing, so a warm cache costs nothing. Every missing member is fetched under a single opened
    archive, since each open re-reads the central directory.
    """
    root = fastlog_fixture_dir(data_dir)
    missing = {day: absent for day, tags in days.items() if (absent := _absent_tags(root, device_id, day, tags))}
    if not missing:
        logger.info("fastlog fixtures already present under %s", root)
        return root

    url = ZENODO_CONTENT_URL.format(record_id=_HOT_V2_RECORD_ID, filename=FASTLOG_ZIP)
    logger.info("fetching fastlog fixture members for %s from %s", sorted(missing), FASTLOG_ZIP)
    with RemoteZip(url) as archive:
        names = archive.namelist()
        for day, tags in missing.items():
            day_dir = _day_dir(root, device_id, day)
            day_dir.mkdir(parents=True, exist_ok=True)
            for member in _members_for(names, device_id=device_id, day=day, tags=tags):
                # Written through a temp name so an interrupted fetch cannot leave a short file
                # that the "already present" check above would then accept on the next run.
                target = day_dir / Path(member).name
                partial = target.with_suffix(target.suffix + ".partial")
                with archive.open(member) as src, partial.open("wb") as dst:
                    dst.write(src.read())
                partial.replace(target)
                logger.info("fetched %s (%d bytes)", target.name, target.stat().st_size)
    return root


def _day_dir(root: Path, device_id: str, day: str) -> Path:
    return root / "FL" / "HOT" / device_id / day


def _absent_tags(root: Path, device_id: str, day: str, tags: Sequence[str]) -> list[str]:
    """Return the tags of ``day`` that are not already on disk."""
    day_dir = _day_dir(root, device_id, day)
    # The `.prq` suffix matters: it is what stops a `.prq.partial` left by an interrupted fetch
    # from reading as present. test_absent_tags_does_not_accept_an_interrupted_partial_file pins it.
    return [tag for tag in tags if not list(day_dir.glob(f"*_{tag}_*.prq"))]


def _members_for(names: Sequence[str], *, device_id: str, day: str, tags: Sequence[str]) -> list[str]:
    """Archive member names for one device-day, one per wanted tag.

    Selected by listing the day's directory rather than by rebuilding the filename convention, so
    a change in that convention surfaces as a clear "not found" rather than a silent miss.
    """
    prefix = f"{_ZIP_FILESTORE_PREFIX}FL/HOT/{device_id}/{day}/"
    in_day = [n for n in names if n.startswith(prefix) and n.endswith(".prq")]
    found = []
    for tag in tags:
        matches = [n for n in in_day if f"_{tag}_" in Path(n).name]
        if not matches:
            msg = (
                f"no fastlog member for tag {tag!r} on {day} for device {device_id} in {FASTLOG_ZIP} "
                f"(record {_HOT_V2_RECORD_ID}); {len(in_day)} file(s) exist for that day"
            )
            raise FileNotFoundError(msg)
        found.extend(matches)
    return found
