import numpy as np
import pandas as pd
import pytest

pytest.importorskip("floris")

from geographiclib.geodesic import Geodesic

from scripts.wedowind_wake_model_2026.toggle_off_t07 import (
    fit_ws_scale,
    load_hot_floris_model,
    local_east_north,
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
    """The bounded fit must recover the wind speed scale used to make the synthetic measurements."""
    rng = np.random.default_rng(0)
    ws_raw = rng.uniform(4, 12, size=200)

    def power_curve(ws: np.ndarray) -> np.ndarray:
        return np.clip(2300 * ((ws - 3) / 9) ** 3, 0, 2300)

    true_k = 0.93
    measured = power_curve(true_k * ws_raw)

    k = fit_ws_scale(modelled_power_fn=lambda k: power_curve(k * ws_raw), measured_power=measured)

    assert k == pytest.approx(true_k, abs=1e-3)
