import numpy as np
import pandas as pd
import pytest

pytest.importorskip("floris")

from floris import FlorisModel

from scripts.wedowind_wake_model_2026.hot_floris import FARM_YAML_PATH, yaw_angles_for, yaw_misalignment
from scripts.wedowind_wake_model_2026.zxtm_probes import (
    CONE_CENTRE_HEIGHT_M,
    HALF_ANGLE_DEG,
    HUB_HEIGHT_M,
    LIDAR_BEHIND_ROTOR_M,
    PROBE_RANGES_M,
    RANGE_COL,
    SEGMENT_HEIGHTS_M,
    SIDES,
    left_minus_right,
    los_col,
    los_magnitude,
    mean_arc_lateral,
    modelled_los_minutes,
    probe_points,
    zxtm_minute_means,
)


def _heights_at(points: pd.DataFrame, range_m: float) -> set[float]:
    return set(points.xs(range_m, level="range_m").index.get_level_values("segment_height_m"))


def test_probes_exist_where_the_measured_data_has_values() -> None:
    """A circle centred 3.2 m above the hub just misses 42.6 m at 64 m range and 26.2 m at 126 m, as the data does.

    The data also has 75.4 m at 48 m range about half the time, when nacelle inclination under load lifts the
    circle the 0.3 m it misses by; that probe is left out.
    """
    points = probe_points(rotor_x=0.0, rotor_y=0.0, yaw_pos_deg=0.0)

    assert _heights_at(points, 48.0) == {59.0}
    assert _heights_at(points, 64.0) == {59.0, 75.4}
    assert _heights_at(points, 85.0) == {42.6, 59.0, 75.4}
    assert _heights_at(points, 126.0) == {42.6, 59.0, 75.4, 91.8}
    assert _heights_at(points, 208.0) == set(SEGMENT_HEIGHTS_M)


def test_one_probe_per_segment_and_side() -> None:
    """The probe table is keyed like the measured minute table, with both sides at every reached segment."""
    points = probe_points(rotor_x=0.0, rotor_y=0.0, yaw_pos_deg=0.0)

    assert points.index.names == ["range_m", "segment_height_m", "side"]
    assert not points.index.duplicated().any()
    assert set(points.index.get_level_values("side")) == set(SIDES)
    assert (points.xs("Left", level="side").index == points.xs("Right", level="side").index).all()


def test_mean_arc_lateral_matches_the_mean_over_scan_angle() -> None:
    """The lateral position is the mean over evenly spaced scan angles of the arc within the segment."""
    radius = 30.0
    angles = np.linspace(-np.pi / 2, np.pi / 2, 200_001)
    for up in (-29.0, -10.0, 0.0, 12.0, 25.0):
        in_segment = np.abs(radius * np.sin(angles) - up) <= 8.2
        expected = (radius * np.cos(angles[in_segment])).mean()
        assert mean_arc_lateral(radius=radius, up=up) == pytest.approx(expected, abs=1e-3)


def test_probe_sits_at_the_mean_lateral_position_of_its_segment() -> None:
    """Facing north, a probe is R - 0.8 m north of the rotor and at its segment's mean arc offset east or west.

    The middle segments sit close to where the circle crosses their height; the top and bottom ones further in.
    """
    points = probe_points(rotor_x=0.0, rotor_y=0.0, yaw_pos_deg=0.0)
    radius = 208.0 * np.tan(np.deg2rad(HALF_ANGLE_DEG))

    for height in SEGMENT_HEIGHTS_M:
        up = height - CONE_CENTRE_HEIGHT_M
        lateral = mean_arc_lateral(radius=radius, up=up)
        crossing = np.sqrt(radius**2 - up**2)
        assert points.loc[(208.0, height, "Right"), "x"] == pytest.approx(lateral)
        assert points.loc[(208.0, height, "Left"), "x"] == pytest.approx(-lateral)
        assert points.loc[(208.0, height, "Right"), "y"] == pytest.approx(208.0 - LIDAR_BEHIND_ROTOR_M)
        assert points.loc[(208.0, height, "Right"), "z"] == pytest.approx(height)
        assert crossing - 1.0 < lateral < crossing
    assert points.loc[(126.0, 91.8, "Right"), "x"] < np.sqrt((126.0 * np.tan(np.deg2rad(15.0))) ** 2 - 29.6**2) - 2.0


def test_right_probe_is_to_the_right_looking_upwind() -> None:
    """Facing north the right probe is east; facing east it is south."""
    north = probe_points(rotor_x=0.0, rotor_y=0.0, yaw_pos_deg=0.0)
    east = probe_points(rotor_x=0.0, rotor_y=0.0, yaw_pos_deg=90.0)

    assert (north.xs("Right", level="side")["x"] > 0).all()
    assert (north.xs("Left", level="side")["x"] < 0).all()
    assert (north["y"] > 0).all()
    assert (east.xs("Right", level="side")["y"] < 0).all()
    assert (east["x"] > 0).all()


def test_beams_are_15_degrees_off_the_axis_from_the_optical_head() -> None:
    """Every mean beam is cos 15 deg along the axis and points from the optical head towards its probe."""
    yaw = np.deg2rad(33.0)
    axis = np.array([np.sin(yaw), np.cos(yaw), 0.0])
    head = np.array([10.0, 20.0, CONE_CENTRE_HEIGHT_M]) - LIDAR_BEHIND_ROTOR_M * axis
    points = probe_points(rotor_x=10.0, rotor_y=20.0, yaw_pos_deg=33.0)
    offset = points[["x", "y", "z"]].to_numpy() - head
    beam = points[["beam_x", "beam_y", "beam_z"]].to_numpy()
    ranges = points.index.get_level_values("range_m").to_numpy()

    np.testing.assert_allclose(offset @ axis, ranges)
    np.testing.assert_allclose(beam @ axis, np.cos(np.deg2rad(HALF_ANGLE_DEG)))
    np.testing.assert_allclose(beam, offset / (ranges / np.cos(np.deg2rad(HALF_ANGLE_DEG)))[:, None])


def test_wind_along_the_axis_gives_u_cos_15_on_every_probe() -> None:
    """With the wind along the cone axis every beam sees the same u cos 15 deg, whatever its height."""
    points = probe_points(rotor_x=0.0, rotor_y=0.0, yaw_pos_deg=200.0)

    los = los_magnitude(u=8.0, wd_deg=200.0, points=points)

    np.testing.assert_allclose(los, 8.0 * np.cos(np.deg2rad(HALF_ANGLE_DEG)))


def test_wind_from_the_right_of_the_axis_reads_faster_on_the_right_beam() -> None:
    """Pins the LOS projection sign: wind from the right of the axis lines up better with the right beam."""
    points = probe_points(rotor_x=0.0, rotor_y=0.0, yaw_pos_deg=0.0)
    los = pd.Series(los_magnitude(u=8.0, wd_deg=10.0, points=points), index=points.index)

    for range_m in (48.0, 85.0, 208.0):
        assert los[(range_m, HUB_HEIGHT_M, "Right")] > los[(range_m, HUB_HEIGHT_M, "Left")]


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
    """FLORIS sampling at the probes gives u cos 15 deg on every one in uniform, unwaked flow."""
    fmodel = FlorisModel(FARM_YAML_PATH)
    fmodel.set(layout_x=[0.0], layout_y=[0.0], wind_shear=0.0)

    los = modelled_los_minutes(
        fmodel, turbine_names=["down"], df=_minute(wd=270.0, lidar_yaw_pos=270.0), k=1.0, lidar_wtg="down"
    )

    assert los.shape[0] == 1
    assert los.columns.names == ["range_m", "segment_height_m", "side"]
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
