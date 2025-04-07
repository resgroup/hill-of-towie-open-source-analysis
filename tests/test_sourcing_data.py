from pathlib import Path

import responses

from hot_open.sourcing_data import download_zenodo_data


class TestDownloadZenodoData:
    @staticmethod
    @responses.activate
    def test_downloads_file_correctly(tmp_path: Path):
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
