"""Tests for the fastlog fixture fetcher's selection logic (no network).

``ensure_fastlog_days`` itself is exercised for real by ``test_fl_reproduces_10min.py``, which
cannot run at all unless the fetch works. What is worth pinning separately is the part with
decisions in it: which archive members answer a wanted tag, and which tags are already on disk.
"""

from pathlib import Path

import pytest

from tests.zenodo_fastlog import _HOT_V2_RECORD_ID, _absent_tags, _members_for, fastlog_fixture_dir

PREFIX = "turbine_fastlog/Filestore/FL/HOT/2304510/2026-03-17/"
NAMES = [
    f"{PREFIX}",
    f"{PREFIX}FL2304510_Wtc_TDI_ActPower_Value_2026_03_17.prq",
    f"{PREFIX}FL2304510_Wtc_TDI_YawPos_Value_2026_03_17.prq",
    f"{PREFIX}FL2304510_Wtc_TDI_AcWindSp_AcWindSp_2026_03_17.prq",
    # A different device and a different day, to prove the prefix filter does its job.
    "turbine_fastlog/Filestore/FL/HOT/2304511/2026-03-17/FL2304511_Wtc_TDI_YawPos_Value_2026_03_17.prq",
    "turbine_fastlog/Filestore/FL/HOT/2304510/2026-04-25/FL2304510_Wtc_TDI_YawPos_Value_2026_04_25.prq",
]


def test_members_for_picks_one_member_per_tag() -> None:
    """Each wanted tag resolves to its own member, in the order asked for."""
    found = _members_for(NAMES, device_id="2304510", day="2026-03-17", tags=["YawPos_Value", "ActPower_Value"])
    assert found == [
        f"{PREFIX}FL2304510_Wtc_TDI_YawPos_Value_2026_03_17.prq",
        f"{PREFIX}FL2304510_Wtc_TDI_ActPower_Value_2026_03_17.prq",
    ]


def test_members_for_ignores_other_devices_and_days() -> None:
    """The prefix filter must not let a neighbouring device or date leak in."""
    found = _members_for(NAMES, device_id="2304510", day="2026-03-17", tags=["YawPos_Value"])
    assert len(found) == 1
    assert "2304511" not in found[0]
    assert "2026-04-25" not in found[0]


def test_members_for_raises_naming_the_tag_it_could_not_find() -> None:
    """A missing tag must be loud: silently fetching nothing would surface much later."""
    with pytest.raises(FileNotFoundError, match="PitcPosA_Value"):
        _members_for(NAMES, device_id="2304510", day="2026-03-17", tags=["PitcPosA_Value"])


def test_absent_tags_reports_everything_missing_when_nothing_is_downloaded(tmp_path: Path) -> None:
    """An empty cache wants every tag."""
    assert _absent_tags(tmp_path, "2304510", "2026-03-17", ["YawPos_Value", "ActPower_Value"]) == [
        "YawPos_Value",
        "ActPower_Value",
    ]


def test_absent_tags_skips_a_tag_already_on_disk(tmp_path: Path) -> None:
    """A warm cache is what keeps the network out of a repeat run."""
    day_dir = tmp_path / "FL" / "HOT" / "2304510" / "2026-03-17"
    day_dir.mkdir(parents=True)
    (day_dir / "FL2304510_Wtc_TDI_YawPos_Value_2026_03_17.prq").write_bytes(b"")
    assert _absent_tags(tmp_path, "2304510", "2026-03-17", ["YawPos_Value", "ActPower_Value"]) == ["ActPower_Value"]


def test_absent_tags_does_not_accept_an_interrupted_partial_file(tmp_path: Path) -> None:
    """A half-written fetch must not read as present, or the next run would use a short file."""
    day_dir = tmp_path / "FL" / "HOT" / "2304510" / "2026-03-17"
    day_dir.mkdir(parents=True)
    (day_dir / "FL2304510_Wtc_TDI_YawPos_Value_2026_03_17.prq.partial").write_bytes(b"")
    assert _absent_tags(tmp_path, "2304510", "2026-03-17", ["YawPos_Value"]) == ["YawPos_Value"]


def test_fixture_dir_is_keyed_on_the_record_id(tmp_path: Path) -> None:
    """A new datapack must miss rather than silently reuse the previous one's files."""
    assert fastlog_fixture_dir(tmp_path).name == f"fl_fixtures_{_HOT_V2_RECORD_ID}"
