from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from wind_up.constants import DataColumns

from hot_open.scada_helpers import calc_shutdown_duration, load_hot_10min_data
from tests.synthetic_scada import SERIAL_OFFSET, field_value, write_year_zips

TIMEBASE_S = 600


class TestLoadHot10minDataSynthetic:
    START = pd.Timestamp("2025-03-01", tz="UTC")
    END_EXCL = pd.Timestamp("2025-03-01 01:00", tz="UTC")

    def test_wide_frame_with_raw_field_names_and_serial_numbers(self, tmp_path: Path) -> None:
        write_year_zips(tmp_path, start_dt=self.START, end_dt_excl=self.END_EXCL, wtg_numbers=[1, 2, 3])
        df = load_hot_10min_data(
            data_dir=tmp_path,
            wtg_numbers=[1, 3],
            start_dt=self.START,
            end_dt_excl=self.END_EXCL,
            use_turbine_names=False,
        )
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "TimeStamp_StartFormat"
        assert str(df.index.tz) == "UTC"
        assert len(df) == 6
        assert df.columns.nlevels == 2
        assert sorted(df.columns.get_level_values(0).unique()) == [1 + SERIAL_OFFSET, 3 + SERIAL_OFFSET]
        assert "wtc_ActPower_mean" in df.columns.get_level_values(1)
        assert df[(3 + SERIAL_OFFSET, "wtc_ActPower_mean")].iloc[0] == field_value(
            field_name="wtc_ActPower_mean", wtg_number=3, start_ts=self.START
        )

    @pytest.mark.parametrize(
        ("start_dt", "end_dt_excl", "match"),
        [
            (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-02", tz="UTC"), "UTC"),
            (pd.Timestamp("2025-03-02", tz="UTC"), pd.Timestamp("2025-03-01", tz="UTC"), "after"),
        ],
    )
    def test_rejects_bad_window(
        self, tmp_path: Path, start_dt: pd.Timestamp, end_dt_excl: pd.Timestamp, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            load_hot_10min_data(data_dir=tmp_path, wtg_numbers=[1], start_dt=start_dt, end_dt_excl=end_dt_excl)

    def test_rejects_unexpected_index_name(self, tmp_path: Path) -> None:
        """A CSV whose first column is not TimeStamp would be misread as end-format times; it must fail loudly."""
        write_year_zips(tmp_path, start_dt=self.START, end_dt_excl=self.END_EXCL, index_name="NotTheTimeStamp")
        with pytest.raises(ValueError, match="unexpected index name"):
            load_hot_10min_data(data_dir=tmp_path, wtg_numbers=[1], start_dt=self.START, end_dt_excl=self.END_EXCL)


def _wind_up_df(power_by_turbine: dict[str, list[float]]) -> pd.DataFrame:
    """Build an interleaved narrow wind_up_df (one row per timestamp and turbine).

    Secondary signals are derived from power so that a frozen power series yields a frozen
    record (stuck), and a varying power series yields a varying record. Rows are sorted by
    timestamp, interleaving the turbines just as ``scada_df_to_wind_up_df`` does.
    """
    n = len(next(iter(power_by_turbine.values())))
    index = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
    index.name = "TimeStamp_StartFormat"
    frames = []
    for turbine, powers in power_by_turbine.items():
        power = np.asarray(powers, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "TurbineName": turbine,
                    DataColumns.active_power_mean: power,
                    DataColumns.active_power_sd: power * 0.01,
                    DataColumns.wind_speed_mean: 5.0 + power / 200.0,  # always > 1.5 m/s threshold
                    DataColumns.wind_speed_sd: 1.0,
                    DataColumns.gen_rpm_mean: 1000.0 + power,
                    DataColumns.pitch_angle_mean: 1.0,
                    DataColumns.yaw_angle_mean: 180.0,
                    "Time ready to operate in period": float(TIMEBASE_S),
                },
                index=index,
            )
        )
    return pd.concat(frames).sort_index(kind="stable")


def test_calc_shutdown_duration_zero_when_turbines_match_but_vary_over_time() -> None:
    """Two turbines that share values per timestamp but vary over time are all available.

    Stuck detection must compare each turbine to its own previous record, so turbines that
    merely match each other at the same timestamp are not flagged as frozen.
    """
    wind_up_df = _wind_up_df({"T01": [100.0, 200.0, 300.0, 400.0], "T02": [100.0, 200.0, 300.0, 400.0]})
    out = calc_shutdown_duration(wind_up_df)
    assert np.allclose(out[DataColumns.shutdown_duration], 0.0)


def test_calc_shutdown_duration_flags_only_the_frozen_turbine() -> None:
    """In a multi-turbine frame, only the turbine whose own data is frozen is flagged."""
    wind_up_df = _wind_up_df({"T01": [100.0, 100.0, 100.0, 100.0], "T02": [100.0, 200.0, 300.0, 400.0]})
    out = calc_shutdown_duration(wind_up_df)
    t01 = out.loc[out["TurbineName"] == "T01", DataColumns.shutdown_duration].to_numpy()
    t02 = out.loc[out["TurbineName"] == "T02", DataColumns.shutdown_duration].to_numpy()
    assert np.allclose(t01[1:], TIMEBASE_S)  # frozen turbine -> downtime after the first row
    assert np.allclose(t02, 0.0)  # varying turbine -> available throughout


def test_calc_shutdown_duration_does_not_flag_low_wind_stuck_data() -> None:
    """Frozen data below the low-wind threshold is not treated as downtime."""
    # Power 0 -> wind speed 5.0 in the helper, so build a low-wind frozen case directly.
    n = 4
    index = pd.date_range("2020-01-01", periods=n, freq="10min", tz="UTC")
    wind_up_df = pd.DataFrame(
        {
            "TurbineName": "T01",
            DataColumns.active_power_mean: 0.0,
            DataColumns.active_power_sd: 0.0,
            DataColumns.wind_speed_mean: 1.0,  # below 1.5 m/s threshold
            DataColumns.wind_speed_sd: 0.0,
            DataColumns.gen_rpm_mean: 0.0,
            DataColumns.pitch_angle_mean: 0.0,
            DataColumns.yaw_angle_mean: 180.0,
            "Time ready to operate in period": float(TIMEBASE_S),
        },
        index=index,
    )
    out = calc_shutdown_duration(wind_up_df)
    assert np.allclose(out[DataColumns.shutdown_duration], 0.0)
