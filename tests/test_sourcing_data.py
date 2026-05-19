import io
import zipfile
from pathlib import Path

import pytest
import responses

from hot_open.sourcing_data import download_zenodo_data, ensure_extracted, ensure_hot_data_files


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
