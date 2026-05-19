from pathlib import Path

import responses

from hot_open.sourcing_data import download_zenodo_data, ensure_hot_data_files


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
