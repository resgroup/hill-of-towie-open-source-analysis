from pathlib import Path

import pandas as pd

from hot_open.helpers import WPSBackupFileField, load_hot_10min_data


def test_load_hot_10min_data()->None:
    custom_fields = [
    WPSBackupFileField(
        alias="Time ready to operate in period", field_name="wtc_ScReToOp_timeon", table_name="tblSCTurFlag"
    ),
        WPSBackupFileField(
            alias="Time running in period", field_name="wtc_ScInOper_timeon", table_name="tblSCTurFlag"
        ),
        WPSBackupFileField(
            alias="Time turbine error active in period", field_name="wtc_ScTurSto_timeon", table_name="tblSCTurFlag"
        ),
    ]
    start_dt = pd.Timestamp("2024-07-31 12:00:00", tz="UTC")
    end_dt_excl = pd.Timestamp("2024-09-02", tz="UTC")
    test_data_dir=Path(__file__).parent / "test_data"
    actual = load_hot_10min_data(data_dir=test_data_dir,wtg_numbers = list(range(1,22)), start_dt = start_dt, end_dt_excl= end_dt_excl,custom_fields=custom_fields)
    assert actual.index.min() == pd.Timestamp("2024-07-31 12:00:00",tz="UTC")
    assert actual.index.max() == pd.Timestamp("2024-08-31 23:50:00", tz="UTC") # last row of data in source file
    assert len(actual) == (actual.index.max() - actual.index.min()).total_seconds()/600 + 1
    assert len(actual.columns) == 3 * 21
    expected=pd.read_parquet(test_data_dir/"test_load_hot_10min_data_expected.parquet")
    pd.testing.assert_frame_equal(actual, expected,check_freq=False) # for some reason freq does not read in correctly
