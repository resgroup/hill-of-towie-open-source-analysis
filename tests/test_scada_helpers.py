import numpy as np
import pandas as pd
from wind_up.constants import DataColumns

from hot_open.scada_helpers import calc_shutdown_duration

TIMEBASE_S = 600


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
