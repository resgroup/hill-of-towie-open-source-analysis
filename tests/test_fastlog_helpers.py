import pandas as pd
import pytest

from hot_open.fastlog_helpers import (
    SIEMENS_PARKS,
    SIEMENS_TAGS,
    _get_raw_df_dict,
    _get_tag_list_from_park_id,
)


class TestSiemensParks:
    def test_default_is_hot_only(self) -> None:
        # Guard against re-leaking private park ids into the public default.
        assert {"HOT"} == SIEMENS_PARKS

    def test_tag_list_for_default_park(self) -> None:
        assert _get_tag_list_from_park_id("HOT") == SIEMENS_TAGS

    def test_tag_list_unknown_park_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            _get_tag_list_from_park_id("EXAMPLE")

    def test_tag_list_injected_park(self) -> None:
        assert _get_tag_list_from_park_id("EXAMPLE", siemens_parks={"EXAMPLE"}) == SIEMENS_TAGS

    def test_raw_df_dict_unknown_park_raises(self, tmp_path: object) -> None:
        with pytest.raises(NotImplementedError):
            _get_raw_df_dict(
                park_id="EXAMPLE",
                device_id="X",
                start_dt=pd.Timestamp("2024-01-01"),
                end_dt_excl=pd.Timestamp("2024-01-02"),
                filestore_dir=tmp_path,  # type: ignore[arg-type]
                tags=["ActPower_Value"],
            )

    def test_raw_df_dict_injected_park_no_data(self, tmp_path: object) -> None:
        result = _get_raw_df_dict(
            park_id="EXAMPLE",
            device_id="X",
            start_dt=pd.Timestamp("2024-01-01"),
            end_dt_excl=pd.Timestamp("2024-01-02"),
            filestore_dir=tmp_path,  # type: ignore[arg-type]
            tags=["ActPower_Value"],
            siemens_parks={"EXAMPLE"},
        )
        assert result["ActPower_Value"].empty
