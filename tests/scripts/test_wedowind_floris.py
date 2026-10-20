import numpy as np
import pandas as pd
import pytest

pytest.importorskip("floris")

from geographiclib.geodesic import Geodesic

from scripts.wedowind_wake_model_2026.toggle_off_t07 import (
    ACT_LIMIT_COL,
    GEN_STATE_COL,
    POWER_COL,
    TOGGLE_COL,
    WD_COL,
    WINDOW_START,
    WS_COL,
    WTGS,
    check_data_present,
    fill_short_direction_gaps,
    filter_minutes,
    fit_ws_scale,
    load_hot_floris_model,
    local_east_north,
    make_minute_df,
    rolling_turbulence_intensity,
    run_floris,
)

# from the Zenodo turbine metadata
T03_LAT_LON = (57.50206973, -3.088980479)
T07_LAT_LON = (57.50513302, -3.085850762)


def _t03_t07_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["T03", "T07"],
            "Latitude": [T03_LAT_LON[0], T07_LAT_LON[0]],
            "Longitude": [T03_LAT_LON[1], T07_LAT_LON[1]],
        }
    )


def test_local_east_north_keeps_t03_to_t07_distance_and_bearing() -> None:
    """The layout must place T07 about 389 m from T03 at a bearing of about 29 deg, as on the ground."""
    x, y = local_east_north(latitudes=[T03_LAT_LON[0], T07_LAT_LON[0]], longitudes=[T03_LAT_LON[1], T07_LAT_LON[1]])
    dx, dy = x[1] - x[0], y[1] - y[0]
    geodesic = Geodesic.WGS84.Inverse(*T03_LAT_LON, *T07_LAT_LON)

    assert np.hypot(dx, dy) == pytest.approx(geodesic["s12"], abs=0.1)
    assert np.degrees(np.arctan2(dx, dy)) % 360 == pytest.approx(geodesic["azi1"] % 360, abs=0.01)
    assert np.hypot(dx, dy) == pytest.approx(389, abs=5)
    assert np.degrees(np.arctan2(dx, dy)) % 360 == pytest.approx(29, abs=2)


def test_public_model_runs_and_waked_turbine_produces_less() -> None:
    """The committed farm and turbine yamls load, give finite power and T07 is waked by T03 from 209 deg."""
    fmodel, turbine_names = load_hot_floris_model(_t03_t07_metadata())
    power = run_floris(
        fmodel,
        turbine_names=turbine_names,
        wind_speeds=[6.0, 8.0, 10.0],
        wind_directions=[209.0, 209.0, 209.0],
        turbulence_intensities=[0.08, 0.08, 0.08],
    )

    assert power.shape == (3, 2)
    assert np.isfinite(power.to_numpy()).all()
    assert (power["T07"] < power["T03"]).all()


def test_fit_ws_scale_recovers_known_factor() -> None:
    """The zero-bias fit must recover the wind speed scale used to make the synthetic measurements."""
    rng = np.random.default_rng(0)
    ws_raw = rng.uniform(4, 12, size=200)

    def power_curve(ws: np.ndarray) -> np.ndarray:
        return np.clip(2300 * ((ws - 3) / 9) ** 3, 0, 2300)

    true_k = 0.93
    measured = power_curve(true_k * ws_raw)

    k = fit_ws_scale(modelled_power_fn=lambda k: power_curve(k * ws_raw), measured_power=measured)

    assert k == pytest.approx(true_k, abs=1e-3)


_PAD = pd.Timedelta(minutes=10)


def _one_hz_fl_df(n_minutes: int) -> pd.DataFrame:
    """1 Hz fastlog for T01, T03 and T07 where every minute passes the filter, at 209 deg.

    Padded by 10 minutes either side so every TI window from ``WINDOW_START`` on is full.
    """
    index = pd.date_range(
        WINDOW_START - _PAD, WINDOW_START + pd.Timedelta(minutes=n_minutes) + _PAD, freq="1s", inclusive="left"
    ).rename("timestamp")
    per_wtg = pd.DataFrame(
        {
            POWER_COL: 800.0,
            WS_COL: np.tile([7.8, 8.2], len(index) // 2),
            GEN_STATE_COL: "LargeOnGrd",
            ACT_LIMIT_COL: 2300.0,
            TOGGLE_COL: 0.0,
            WD_COL: 209.0,
        },
        index=index,
    )
    return pd.concat({wtg: per_wtg.copy() for wtg in WTGS}, axis=1, names=["device_id", "tag"])


def _unpadded_minute_df(fl_df: pd.DataFrame) -> pd.DataFrame:
    minute_df = make_minute_df(fl_df)
    return minute_df.loc[(minute_df.index >= fl_df.index[0] + _PAD) & (minute_df.index < fl_df.index[-1] - _PAD)]


def test_filter_minutes_applies_each_selection_criterion() -> None:
    """Each minute below breaks one criterion; a regression in any of them changes which minutes are kept."""
    fl_df = _one_hz_fl_df(n_minutes=10)

    def second_in(minute: int, second: int = 30) -> pd.Timestamp:
        return WINDOW_START + pd.Timedelta(minutes=minute, seconds=second)

    fl_df.loc[second_in(1), ("T03", TOGGLE_COL)] = 1.0  # one toggle-on second
    fl_df.loc[second_in(2), ("T07", TOGGLE_COL)] = np.nan  # toggle state missing for a second
    fl_df.loc[second_in(3), ("T07", GEN_STATE_COL)] = "GeneratOut"  # off grid for a second
    fl_df.loc[second_in(4), ("T01", ACT_LIMIT_COL)] = 2000.0  # curtailed for a second
    fl_df.loc[second_in(5), ("T03", POWER_COL)] = -5.0  # not producing for a second
    minute_wd = {6: 192.2, 7: 191.8, 8: 223.8, 9: 224.2}
    for minute, wd in minute_wd.items():
        in_minute = (fl_df.index >= second_in(minute, 0)) & (fl_df.index < second_in(minute + 1, 0))
        fl_df.loc[in_minute, ("T01", WD_COL)] = wd

    kept = filter_minutes(_unpadded_minute_df(fl_df))

    expected_minutes = [0, 6, 8]
    assert list(kept.index) == [second_in(m, 0) for m in expected_minutes]
    assert kept["ti"].between(0.03, 0.25).all()


def test_filter_minutes_raises_with_counts_when_nothing_survives() -> None:
    """An empty selection must fail loudly and say which filter step removed the minutes."""
    fl_df = _one_hz_fl_df(n_minutes=3)
    fl_df[("T03", TOGGLE_COL)] = 1.0

    with pytest.raises(ValueError, match="toggle off all minute': 3"):
        filter_minutes(_unpadded_minute_df(fl_df))


def test_check_data_present_names_turbine_and_day() -> None:
    """A day with no power for one turbine must be a hard error naming that turbine and day."""
    index = pd.date_range(WINDOW_START, periods=3, freq="D", name="timestamp")
    fl_df = pd.concat(
        {wtg: pd.DataFrame({POWER_COL: [800.0, 800.0, 800.0]}, index=index) for wtg in WTGS},
        axis=1,
    )
    fl_df.loc[index[1], ("T07", POWER_COL)] = np.nan

    with pytest.raises(ValueError, match="T07 on 2026-02-21"):
        check_data_present(fl_df, wtgs=WTGS, start_dt=index[0], end_dt_excl=index[-1] + pd.Timedelta(days=1))


def test_fill_short_direction_gaps_wraps_north_and_skips_long_gaps() -> None:
    """A short gap across north must fill the short way round; a gap of 10 minutes or more stays NaN."""
    index = pd.date_range(WINDOW_START, periods=30, freq="1min")
    wd = pd.Series(np.nan, index=index)
    wd.iloc[0], wd.iloc[4] = 350.0, 10.0  # 3-minute gap across north
    wd.iloc[5:10] = 20.0
    wd.iloc[20:] = 40.0  # 10-minute gap from minute 10 to 19

    filled = fill_short_direction_gaps(wd)

    np.testing.assert_allclose(filled.iloc[:5], [350.0, 355.0, 0.0, 5.0, 10.0], atol=1e-9)
    assert filled.iloc[10:20].isna().all()
    assert (filled.iloc[20:] == 40.0).all()


def test_rolling_turbulence_intensity_is_ten_minute_std_over_mean() -> None:
    """TI must be the 1 Hz std over mean of the centred 10-minute window, not of a single minute."""
    index = pd.date_range(WINDOW_START, periods=30 * 60, freq="1s")
    ws = pd.Series(np.where(np.arange(len(index)) // 60 % 2 == 0, 7.0, 9.0), index=index)  # alternates each minute
    at = pd.DatetimeIndex([WINDOW_START + pd.Timedelta(minutes=15)])

    ti = rolling_turbulence_intensity(ws, at=at)

    window = ws[(ws.index > at[0] - pd.Timedelta(minutes=5)) & (ws.index <= at[0] + pd.Timedelta(minutes=5))]
    assert ti.iloc[0] == pytest.approx(window.std() / window.mean())
    assert ti.iloc[0] == pytest.approx(1.0 / 8.0, rel=0.01)
