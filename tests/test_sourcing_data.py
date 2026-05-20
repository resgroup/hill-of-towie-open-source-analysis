import io
import json
import zipfile
from pathlib import Path

import pytest
import requests
import responses

from hot_open import sourcing_data
from hot_open.sourcing_data import download_zenodo_data, ensure_extracted, ensure_hot_data_files


@pytest.fixture(autouse=True)
def _no_sleep_between_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the exponential backoff sleeps so the suite stays fast.

    `_download_one_file` retries up to 5 times with up-to-16s sleeps between
    attempts; without this, tests that exercise the failure paths would add
    30s of real wall-clock waits each.
    """
    monkeypatch.setattr(sourcing_data.time, "sleep", lambda _seconds: None)


class TestDownloadZenodoData:
    @staticmethod
    @responses.activate
    def test_downloads_file_correctly(tmp_path: Path) -> None:
        fname = "somefile.zip"
        expected_fpath = tmp_path / fname
        expected_content = b"test"

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={"files": [{"key": fname, "links": {"self": "http://myfile.url"}, "size": 1}]},
            )
        )
        responses.add(responses.Response(method="GET", url="http://myfile.url", body=expected_content))

        download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=[fname])

        assert expected_fpath.exists()
        with expected_fpath.open("rb") as f:
            downloaded_content = f.read()
        assert downloaded_content == expected_content

    @staticmethod
    @responses.activate
    def test_includes_small_files_alongside_requested_file(tmp_path: Path) -> None:
        big_fname = "big.zip"
        small_fname = "README.md"
        big_size = 3 * 1024 * 1024  # 3 MB - over the 2 MB threshold
        small_content = b"# README\n"

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={
                    "files": [
                        {"key": big_fname, "links": {"self": "http://big.url"}, "size": big_size},
                        {"key": small_fname, "links": {"self": "http://small.url"}, "size": len(small_content)},
                    ]
                },
            )
        )
        responses.add(responses.Response(method="GET", url="http://big.url", body=b"BIGFILE"))
        responses.add(responses.Response(method="GET", url="http://small.url", body=small_content))

        download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=[big_fname])

        assert (tmp_path / big_fname).is_file()
        assert (tmp_path / small_fname).read_bytes() == small_content

    @staticmethod
    @responses.activate
    def test_threshold_is_strictly_under_2mb(tmp_path: Path) -> None:
        # boundary.bin claims exactly 2 MB -> must NOT be auto-included
        # just_under.bin claims 2 MB - 1 byte -> must be auto-included
        boundary_size = 2 * 1024 * 1024
        just_under_size = boundary_size - 1

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={
                    "files": [
                        {"key": "big.zip", "links": {"self": "http://big.url"}, "size": 3 * 1024 * 1024},
                        {"key": "boundary.bin", "links": {"self": "http://boundary.url"}, "size": boundary_size},
                        {"key": "just_under.bin", "links": {"self": "http://just-under.url"}, "size": just_under_size},
                    ]
                },
            )
        )
        responses.add(responses.Response(method="GET", url="http://big.url", body=b"BIG"))
        responses.add(responses.Response(method="GET", url="http://just-under.url", body=b"under"))
        # No mock for http://boundary.url — if the code tries to fetch it, `responses` raises.

        download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=["big.zip"])

        assert (tmp_path / "big.zip").is_file()
        assert (tmp_path / "just_under.bin").read_bytes() == b"under"
        assert not (tmp_path / "boundary.bin").exists()

    @staticmethod
    @responses.activate
    def test_optional_small_file_failure_warns_and_continues(
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        big_fname = "big.zip"
        small_fname = "README.md"

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={
                    "files": [
                        {"key": big_fname, "links": {"self": "http://big.url"}, "size": 3 * 1024 * 1024},
                        {"key": small_fname, "links": {"self": "http://small.url"}, "size": 9},
                    ]
                },
            )
        )
        responses.add(responses.Response(method="GET", url="http://big.url", body=b"BIGFILE"))
        responses.add(
            responses.Response(
                method="GET",
                url="http://small.url",
                body=requests.ConnectionError("offline"),
            )
        )

        with caplog.at_level("WARNING", logger="hot_open.sourcing_data"):
            download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=[big_fname])

        assert (tmp_path / big_fname).is_file()
        assert not (tmp_path / small_fname).exists()
        assert any(small_fname in rec.message for rec in caplog.records if rec.levelname == "WARNING")

    @staticmethod
    @responses.activate
    def test_required_file_failure_still_raises(tmp_path: Path) -> None:
        fname = "README.md"
        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={"files": [{"key": fname, "links": {"self": "http://readme.url"}, "size": 9}]},
            )
        )
        responses.add(
            responses.Response(
                method="GET",
                url="http://readme.url",
                body=requests.ConnectionError("offline"),
            )
        )

        with pytest.raises(requests.RequestException):
            download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=[fname])

        # Required-file partial bytes are intentionally NOT deleted on failure so
        # a subsequent run can resume via Range. Here no bytes were ever written
        # (ConnectionError raised before the first chunk), so the file does not
        # exist.
        assert not (tmp_path / fname).exists()

    @staticmethod
    @responses.activate
    def test_retries_on_transient_connection_error(tmp_path: Path) -> None:
        fname = "somefile.zip"
        expected_content = b"recovered after retry"

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={"files": [{"key": fname, "links": {"self": "http://myfile.url"}, "size": len(expected_content)}]},
            )
        )
        # First call to the file URL raises ConnectionError; second call returns
        # the real bytes. `responses` consumes registered responses in FIFO order.
        responses.add(
            responses.Response(method="GET", url="http://myfile.url", body=requests.ConnectionError("flaky")),
        )
        responses.add(responses.Response(method="GET", url="http://myfile.url", body=expected_content))

        download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=[fname])

        assert (tmp_path / fname).read_bytes() == expected_content

    @staticmethod
    @responses.activate
    def test_required_file_fails_after_max_attempts(tmp_path: Path) -> None:
        fname = "somefile.zip"
        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={"files": [{"key": fname, "links": {"self": "http://myfile.url"}, "size": 100}]},
            )
        )
        # Queue _MAX_DOWNLOAD_ATTEMPTS consecutive ConnectionErrors so every
        # retry attempt observes a transient failure.
        for _ in range(sourcing_data._MAX_DOWNLOAD_ATTEMPTS):  # noqa: SLF001
            responses.add(
                responses.Response(
                    method="GET",
                    url="http://myfile.url",
                    body=requests.ConnectionError("offline"),
                )
            )

        with pytest.raises(requests.RequestException):
            download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=[fname])

        # All MAX attempts were exercised.
        file_calls = [c for c in responses.calls if (c.request.url or "").startswith("http://myfile.url")]
        assert len(file_calls) == sourcing_data._MAX_DOWNLOAD_ATTEMPTS  # noqa: SLF001

    @staticmethod
    @responses.activate
    def test_resume_with_range_header(tmp_path: Path) -> None:
        fname = "somefile.zip"
        prefix = b"abcd"  # pretend a prior attempt got this far
        suffix = b"efghij"
        full_content = prefix + suffix

        # Seed a partial file so the next download attempt resumes from byte 4.
        (tmp_path / fname).write_bytes(prefix)

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/fake-id",
                json={"files": [{"key": fname, "links": {"self": "http://myfile.url"}, "size": len(full_content)}]},
            )
        )
        # Server honors Range: returns only the missing suffix with 206.
        responses.add(
            responses.Response(
                method="GET",
                url="http://myfile.url",
                body=suffix,
                status=206,
                headers={"Content-Range": f"bytes {len(prefix)}-{len(full_content) - 1}/{len(full_content)}"},
            )
        )

        download_zenodo_data(record_id="fake-id", output_dir=tmp_path, filenames=[fname])

        # File on disk is the concatenation of seeded prefix + downloaded suffix.
        assert (tmp_path / fname).read_bytes() == full_content
        # The Range header was actually sent.
        file_calls = [c for c in responses.calls if (c.request.url or "").startswith("http://myfile.url")]
        assert len(file_calls) == 1
        assert file_calls[0].request.headers.get("Range") == f"bytes={len(prefix)}-"


class TestEnsureHotDataFiles:
    @staticmethod
    @responses.activate
    def test_downloads_missing_file(tmp_path: Path) -> None:
        fname = "Hill_of_Towie_turbine_metadata.csv"
        expected_content = b"Name,Latitude,Longitude\n"

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/20204946",
                json={"files": [{"key": fname, "links": {"self": "http://myfile.url"}, "size": len(expected_content)}]},
            )
        )
        responses.add(responses.Response(method="GET", url="http://myfile.url", body=expected_content))

        ensure_hot_data_files([fname], data_dir=tmp_path)

        assert (tmp_path / fname).read_bytes() == expected_content

    @staticmethod
    @responses.activate
    def test_is_noop_when_file_already_present(tmp_path: Path) -> None:
        fname = "Hill_of_Towie_turbine_metadata.csv"
        (tmp_path / fname).write_bytes(b"already here")

        # No HTTP responses registered. If the function attempts any HTTP call,
        # `responses` will raise ConnectionError.
        ensure_hot_data_files([fname], data_dir=tmp_path)

        assert (tmp_path / fname).read_bytes() == b"already here"

    @staticmethod
    @responses.activate
    def test_backfills_small_files_using_cached_metadata(tmp_path: Path) -> None:
        big_fname = "big.zip"
        small_fname = "README.md"
        small_content = b"# README\n"

        # big.zip already present locally
        (tmp_path / big_fname).write_bytes(b"BIGFILE")

        # Pre-seed the metadata cache as if a prior download had written it.
        # No metadata URL is mocked here -- if download_zenodo_data tried to
        # refetch metadata, `responses` would raise ConnectionError.
        metadata_fpath = tmp_path / "zenodo_dataset_metadata.json"
        metadata_fpath.write_text(
            json.dumps(
                {
                    "files": [
                        {"key": big_fname, "links": {"self": "http://big.url"}, "size": 3 * 1024 * 1024},
                        {"key": small_fname, "links": {"self": "http://small.url"}, "size": len(small_content)},
                    ]
                }
            )
        )

        responses.add(responses.Response(method="GET", url="http://small.url", body=small_content))

        ensure_hot_data_files([big_fname], data_dir=tmp_path)

        assert (tmp_path / big_fname).read_bytes() == b"BIGFILE"  # untouched
        assert (tmp_path / small_fname).read_bytes() == small_content


def _build_lidar_zip_bytes() -> bytes:
    """Return zip bytes mimicking the Zenodo lidar_data.zip layout."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("lidar_data/timeseries/2428/Wind10_2428@Y2026_M01_D01.parquet", b"parquet-bytes")
    return buf.getvalue()


class TestEnsureExtracted:
    @staticmethod
    @responses.activate
    def test_downloads_extracts_and_deletes_zip(tmp_path: Path) -> None:
        zip_name = "lidar_data.zip"
        zip_bytes = _build_lidar_zip_bytes()

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/20204946",
                json={"files": [{"key": zip_name, "links": {"self": "http://lidar.zip.url"}, "size": len(zip_bytes)}]},
            )
        )
        responses.add(responses.Response(method="GET", url="http://lidar.zip.url", body=zip_bytes))

        result = ensure_extracted(zip_name, data_dir=tmp_path)

        assert result == tmp_path / "lidar_data"
        assert (tmp_path / "lidar_data" / "timeseries" / "2428" / "Wind10_2428@Y2026_M01_D01.parquet").is_file()
        assert not (tmp_path / zip_name).exists()  # zip removed after extraction

    @staticmethod
    @responses.activate
    def test_is_noop_when_already_extracted(tmp_path: Path) -> None:
        # Pre-create the sentinel layout
        (tmp_path / "lidar_data" / "timeseries").mkdir(parents=True)

        # No HTTP mocks registered: any network call raises.
        result = ensure_extracted("lidar_data.zip", data_dir=tmp_path)

        assert result == tmp_path / "lidar_data"

    @staticmethod
    @responses.activate
    def test_extracts_when_top_level_dir_exists_but_sentinel_missing(tmp_path: Path) -> None:
        """Regression: empty top-level dir left by an unrelated mkdir must not fool the sentinel check.

        For example ``get_filestore_dir()`` eagerly creates ``turbine_fastlog/Filestore/`` even
        when the fastlog data has not been extracted yet. The sentinel must be deeper than that.
        """
        zip_name = "lidar_data.zip"
        zip_bytes = _build_lidar_zip_bytes()

        # Simulate a previous shallow mkdir creating the top-level dir without contents.
        (tmp_path / "lidar_data").mkdir()

        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/20204946",
                json={"files": [{"key": zip_name, "links": {"self": "http://lidar.zip.url"}, "size": len(zip_bytes)}]},
            )
        )
        responses.add(responses.Response(method="GET", url="http://lidar.zip.url", body=zip_bytes))

        result = ensure_extracted(zip_name, data_dir=tmp_path)

        assert result == tmp_path / "lidar_data"
        assert (tmp_path / "lidar_data" / "timeseries" / "2428" / "Wind10_2428@Y2026_M01_D01.parquet").is_file()
        assert not (tmp_path / zip_name).exists()

    @staticmethod
    @responses.activate
    def test_cleans_up_partial_extraction_on_failure(tmp_path: Path) -> None:
        zip_name = "lidar_data.zip"
        # Write garbage as the "zip" so ZipFile() will fail.
        responses.add(
            responses.Response(
                method="GET",
                url="https://zenodo.org/api/records/20204946",
                json={"files": [{"key": zip_name, "links": {"self": "http://garbage.url"}, "size": 4}]},
            )
        )
        responses.add(responses.Response(method="GET", url="http://garbage.url", body=b"junk"))

        with pytest.raises(zipfile.BadZipFile):
            ensure_extracted(zip_name, data_dir=tmp_path)

        # Sentinel must not exist after a failed extract so a re-run will try again.
        assert not (tmp_path / "lidar_data").exists()
