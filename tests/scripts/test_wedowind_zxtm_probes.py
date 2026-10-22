import numpy as np
import pandas as pd
import pytest

pytest.importorskip("floris")

from floris import FlorisModel

from scripts.wedowind_wake_model_2026.hot_floris import FARM_YAML_PATH, yaw_angles_for, yaw_misalignment
from scripts.wedowind_wake_model_2026.zxtm_probes import (
    HALF_ANGLE_DEG,
    HUB_HEIGHT_M,
    PROBE_RANGES_M,
    RANGE_COL,
    SEGMENT_HEIGHTS_M,
    SIDES,
    cone_points,
    left_minus_right,
    los_col,
    los_magnitude,
    modelled_los_minutes,
    segment_means,
    zxtm_minute_means,
)


def _segments_at(points: pd.DataFrame, range_m: float) -> set[float]:
    return set(points.loc[points["range_m"] == range_m, "segment_height_m"].dropna())


def test_cone_reaches_segment_bands_as_in_the_measured_data() -> None:
    """26.2 m only appears once the cone dips below 34.4 m; 75.4 m appears at 48 m because it is a band."""
    points = cone_points(apex_x=0.0, apex_y=0.0, yaw_pos_deg=0.0)

    assert 26.2 not in _segments_at(points, 85.0)
    assert 26.2 in _segments_at(points, 208.0)
    assert 75.4 in _segments_at(points, 48.0)
    assert 91.8 not in _segments_at(points, 48.0)


def test_cone_right_half_is_to_the_right_looking_upwind() -> None:
    """Facing north the right half is east; facing east it is south."""
    north = cone_points(apex_x=0.0, apex_y=0.0, yaw_pos_deg=0.0)
    east = cone_points(apex_x=0.0, apex_y=0.0, yaw_pos_deg=90.0)

    assert (north.loc[north["side"] == "Right", "x"] > 0).all()
    assert (north.loc[north["side"] == "Left", "x"] < 0).all()
    assert (north["y"] > 0).all()
    assert (east.loc[east["side"] == "Right", "y"] < 0).all()
    assert (east["x"] > 0).all()


def test_cone_points_lie_on_a_15_degree_cone_from_the_hub() -> None:
    """Every scan point is range / cos 15 deg from the hub and its beam unit vector points at it."""
    points = cone_points(apex_x=10.0, apex_y=20.0, yaw_pos_deg=33.0)
    offset = np.column_stack([points["x"] - 10.0, points["y"] - 20.0, points["z"] - HUB_HEIGHT_M])
    distance = np.linalg.norm(offset, axis=1)
    beam = points[["beam_x", "beam_y", "beam_z"]].to_numpy()

    np.testing.assert_allclose(distance, points["range_m"] / np.cos(np.deg2rad(HALF_ANGLE_DEG)))
    np.testing.assert_allclose(beam, offset / distance[:, None])


def test_wind_along_the_axis_gives_u_cos_15_on_every_segment() -> None:
    """With the wind along the cone axis every beam sees the same u cos 15 deg, whatever its height."""
    points = cone_points(apex_x=0.0, apex_y=0.0, yaw_pos_deg=200.0)
    means = segment_means(points, los_magnitude(u=8.0, wd_deg=200.0, points=points))

    np.testing.assert_allclose(means.to_numpy(), 8.0 * np.cos(np.deg2rad(HALF_ANGLE_DEG)))
    assert set(means.index.get_level_values("side")) == {"Left", "Right"}


def test_wind_from_the_right_of_the_axis_reads_faster_on_the_right_beam() -> None:
    """Pins the LOS projection sign: wind from the right of the axis lines up better with the right beam."""
    points = cone_points(apex_x=0.0, apex_y=0.0, yaw_pos_deg=0.0)
    means = segment_means(points, los_magnitude(u=8.0, wd_deg=10.0, points=points))

    for range_m in (48.0, 85.0, 208.0):
        assert means[(range_m, HUB_HEIGHT_M, "Right")] > means[(range_m, HUB_HEIGHT_M, "Left")]


def _minute(*, wd: float, lidar_yaw_pos: float, up_yaw_pos: float = 270.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ws_raw": [8.0],
            "wd": [wd],
            "ti": [0.08],
            "down_yaw_position": [lidar_yaw_pos],
            "up_yaw_misalignment": yaw_misalignment(yaw_pos=pd.Series([up_yaw_pos]), wd=pd.Series([wd])).to_numpy(),
        },
        index=pd.DatetimeIndex([pd.Timestamp("2026-02-20 01:00", tz="UTC")], name="timestamp"),
    )


def test_unwaked_lidar_in_uniform_flow_reads_u_cos_15() -> None:
    """FLORIS sampling on the cone gives u cos 15 deg on every segment in uniform, unwaked flow."""
    fmodel = FlorisModel(FARM_YAML_PATH)
    fmodel.set(layout_x=[0.0], layout_y=[0.0], wind_shear=0.0)

    los = modelled_los_minutes(
        fmodel, turbine_names=["down"], df=_minute(wd=270.0, lidar_yaw_pos=270.0), k=1.0, lidar_wtg="down"
    )

    assert los.shape[0] == 1
    np.testing.assert_allclose(los.to_numpy(), 8.0 * np.cos(np.deg2rad(HALF_ANGLE_DEG)), rtol=1e-6)


def test_yawed_upstream_turbine_slows_the_beam_its_wake_deflects_to() -> None:
    """A yawed upstream turbine must slow the beam on the side its wake deflects to.

    Wind from 270 deg, lidar turbine facing west, so its left beam is south. Upstream yaw position 260 deg
    pushes the wake south, so the left LOS drops and the right LOS rises against zero yaw.
    """
    fmodel = FlorisModel(FARM_YAML_PATH)
    fmodel.set(layout_x=[0.0, 400.0], layout_y=[0.0, 0.0])
    turbine_names = ["up", "down"]
    df = _minute(wd=270.0, lidar_yaw_pos=270.0, up_yaw_pos=260.0)

    yawed = modelled_los_minutes(
        fmodel,
        turbine_names=turbine_names,
        df=df,
        k=1.0,
        yaw_angles=yaw_angles_for(df=df, turbine_names=turbine_names, wtg="up"),
        lidar_wtg="down",
    ).iloc[0]
    unyawed = modelled_los_minutes(fmodel, turbine_names=turbine_names, df=df, k=1.0, lidar_wtg="down").iloc[0]

    for range_m in (126.0, 208.0):
        assert yawed[(range_m, HUB_HEIGHT_M, "Left")] < unyawed[(range_m, HUB_HEIGHT_M, "Left")]
        assert yawed[(range_m, HUB_HEIGHT_M, "Right")] > unyawed[(range_m, HUB_HEIGHT_M, "Right")]


def test_zxtm_minute_means_need_five_samples_per_range() -> None:
    """Each range comes round every 6 s, about 10 samples a minute; a probe-minute with fewer than 5 is NaN."""
    index = pd.date_range("2026-02-20 01:00", periods=120, freq="1s", tz="UTC")
    zx_df = pd.DataFrame(
        {los_col(side, h): 6.0 for side in SIDES for h in SEGMENT_HEIGHTS_M}
        | {RANGE_COL: np.tile([10, 48, 64, 85, 126, 208], 20)},
        index=index,
    )
    sparse = (zx_df.index >= index[60]) & (zx_df[RANGE_COL] == 48)
    left_hub = zx_df[los_col("Left", 59.0)].to_numpy()
    left_hub[np.flatnonzero(sparse)[4:]] = np.nan
    zx_df[los_col("Left", 59.0)] = left_hub
    zx_df.index = zx_df.index + pd.Timedelta(milliseconds=300)

    means = zxtm_minute_means(zx_df)

    assert list(means.index) == list(index[[0, 60]])
    assert means.loc[index[0], (48.0, 59.0, "Left")] == pytest.approx(6.0)
    assert means[(48.0, 59.0, "Left")].isna().tolist() == [False, True]
    assert means.loc[index[60], (48.0, 59.0, "Right")] == pytest.approx(6.0)
    assert set(means.columns.get_level_values("range_m")) == set(PROBE_RANGES_M)


def test_left_minus_right_at_hub_height_per_range() -> None:
    """The steering signature uses hub-height segments only, left minus right."""
    columns = pd.MultiIndex.from_tuples(
        [(48.0, 59.0, "Left"), (48.0, 59.0, "Right"), (48.0, 42.6, "Left"), (48.0, 42.6, "Right")],
        names=["range_m", "segment_height_m", "side"],
    )
    df = pd.DataFrame([[7.0, 5.0, 1.0, 9.0]], columns=columns)

    assert left_minus_right(df)[48.0].iloc[0] == pytest.approx(2.0)
