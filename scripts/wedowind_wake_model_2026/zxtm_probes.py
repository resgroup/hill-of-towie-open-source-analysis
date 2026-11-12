"""ZXTM nacelle LiDAR probes on T07 for the public FLORIS model of HOT.

ZXTM geometry and processing, as confirmed by ZX Lidars:

- The ZXTM scans a cone of 15 deg half-angle about T07's rotor axis. It corrects its measurements, the
  high-resolution data included, to zero inclination and roll, so the cone axis is horizontal along T07's
  yaw position.
- The scan circle is centred on the lidar's optical head, 3.2 m above the hub and 0.8 m behind the rotor plane
  (deployment report). ``Range (m)`` is the distance along the cone axis from the optical head, so the scan
  circle at range R has radius R tan 15 deg and is R - 0.8 m ahead of the rotor. The FLORIS turbine position is
  taken as the rotor centre.
- Rotor segment heights are heights above ground; the rotor spans five 16.4 m segments centred on 59 m. Each
  ``Left`` / ``Right`` LOS speed is interpolated to exactly its segment height from the scan samples in that
  segment, on that side of the axis. Its lateral position is the mean of those samples' positions; samples are
  taken as evenly spaced in scan angle, so this is the mean lateral offset of the arc of the scan circle
  within the segment. That is close to where the circle crosses the segment height for the middle segments,
  and further inboard for the top and bottom ones.
- A segment whose height the circle does not reach has no data. Inclination of the nacelle under load moves
  the circle, so a segment near the edge of the circle sometimes has data; 75.4 m at 48 m range, which the
  level circle misses by 0.3 m, has data about half the time. Those segments are not modelled.
- The ZXTM is a continuous-wave lidar, so it reports the magnitude of the LOS speed.

The modelled LOS speed is FLORIS's streamwise speed, along the free-stream wind direction, projected onto
each probe's mean beam; FLORIS's cross-flow from wake deflection is ignored. T07 itself is at zero yaw in
FLORIS while the cone follows T07's measured yaw position. Empirical Gauss has no induction zone, so the
modelled 48 m and 64 m probes miss the slow-down in front of T07.

Run with ``uv sync --group wedowind`` first.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from floris import FlorisModel, TimeSeries

from hot_open.lidar_helpers import load_zx_lidar_fl_data
from hot_open.settings import get_data_dir
from scripts.wedowind_wake_model_2026.hot_floris import PREDICTION_WTG

ZXTM_UNIT_ID = "5060"
HALF_ANGLE_DEG = 15.0
HUB_HEIGHT_M = 59.0
LIDAR_ABOVE_HUB_M = 3.2
LIDAR_BEHIND_ROTOR_M = 0.8
CONE_CENTRE_HEIGHT_M = HUB_HEIGHT_M + LIDAR_ABOVE_HUB_M
SEGMENT_HEIGHTS_M = (26.2, 42.6, 59.0, 75.4, 91.8)
SEGMENT_DEPTH_M = 16.4
PROBE_RANGES_M = (48.0, 64.0, 85.0, 126.0, 208.0)
SIDES = ("Left", "Right")
MIN_SAMPLES_PER_MINUTE = 5
RANGE_COL = "Range (m)"
SEGMENT_LEVELS = ["range_m", "segment_height_m", "side"]


def los_col(side: str, height_m: float) -> str:
    return f"{side} LOS Speed (m/s) at Rotor Segment Height {height_m:.1f}m"


def mean_arc_lateral(*, radius: npt.ArrayLike, up: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the mean lateral offset of one side of the scan circle within the segment centred ``up`` above its centre.

    The segment spans ``up`` +/- half ``SEGMENT_DEPTH_M``, clipped to the circle; the mean is over scan angle.
    """
    r = np.asarray(radius, dtype=float)
    lo = np.clip(np.asarray(up, dtype=float) - SEGMENT_DEPTH_M / 2, -r, r)
    hi = np.clip(np.asarray(up, dtype=float) + SEGMENT_DEPTH_M / 2, -r, r)
    swept = np.arcsin(hi / r) - np.arcsin(lo / r)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(swept > 0, (hi - lo) / swept, np.nan)


def probe_points(*, rotor_x: float, rotor_y: float, yaw_pos_deg: float) -> pd.DataFrame:
    """Return one probe per (range, segment height, side) the scan circle reaches, in farm coordinates.

    ``rotor_x`` / ``rotor_y`` is the lidar turbine's rotor centre and ``yaw_pos_deg`` the compass direction the
    rotor faces. Each probe carries its mean beam vector, whose component along the cone axis is cos 15 deg.
    """
    index = pd.MultiIndex.from_product([PROBE_RANGES_M, SEGMENT_HEIGHTS_M, SIDES], names=SEGMENT_LEVELS)
    ranges = index.get_level_values("range_m").to_numpy(dtype=float)
    heights = index.get_level_values("segment_height_m").to_numpy(dtype=float)
    right = np.where(index.get_level_values("side") == "Right", 1.0, -1.0)
    radius = ranges * np.tan(np.deg2rad(HALF_ANGLE_DEG))
    up = heights - CONE_CENTRE_HEIGHT_M
    reached = np.abs(up) <= radius
    lateral = right * mean_arc_lateral(radius=radius, up=up)
    ahead = ranges - LIDAR_BEHIND_ROTOR_M
    yaw = np.deg2rad(yaw_pos_deg)
    distance = ranges / np.cos(np.deg2rad(HALF_ANGLE_DEG))
    points = pd.DataFrame(
        {
            "x": rotor_x + ahead * np.sin(yaw) + lateral * np.cos(yaw),
            "y": rotor_y + ahead * np.cos(yaw) - lateral * np.sin(yaw),
            "z": heights,
            "beam_x": (ranges * np.sin(yaw) + lateral * np.cos(yaw)) / distance,
            "beam_y": (ranges * np.cos(yaw) - lateral * np.sin(yaw)) / distance,
            "beam_z": up / distance,
        },
        index=index,
    )
    return points.loc[reached]


def los_magnitude(*, u: npt.ArrayLike, wd_deg: float, points: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return |LOS| at ``points`` for a horizontal wind of speed ``u`` blowing from ``wd_deg``."""
    wd = np.deg2rad(wd_deg)
    along_beam = -np.sin(wd) * points["beam_x"].to_numpy() - np.cos(wd) * points["beam_y"].to_numpy()
    return np.abs(np.asarray(u, dtype=float) * along_beam)


def modelled_los_minutes(
    fmodel: FlorisModel,
    *,
    turbine_names: list[str],
    df: pd.DataFrame,
    k: float,
    yaw_angles: npt.ArrayLike | None = None,
    lidar_wtg: str = PREDICTION_WTG,
) -> pd.DataFrame:
    """Return modelled |LOS| per minute of ``df``, one column per (range, segment height, side).

    ``df`` has the free-stream columns and ``{lidar_wtg}_yaw_position``; ``yaw_angles`` is as for ``run_floris``.
    FLORIS runs one minute at a time because the probes move with the lidar turbine's yaw.
    """
    yaw = np.zeros((len(df), len(turbine_names))) if yaw_angles is None else np.asarray(yaw_angles, dtype=float)
    lidar = turbine_names.index(lidar_wtg)
    rotor_x, rotor_y = float(fmodel.layout_x[lidar]), float(fmodel.layout_y[lidar])
    rows = {}
    for i, (timestamp, row) in enumerate(df.iterrows()):
        fmodel.set(
            wind_data=TimeSeries(
                wind_directions=np.array([row["wd"]]),
                wind_speeds=np.array([k * max(row["ws_raw"], 0.0)]),
                turbulence_intensities=np.array([row["ti"]]),
            ),
            yaw_angles=yaw[i : i + 1],
        )
        points = probe_points(rotor_x=rotor_x, rotor_y=rotor_y, yaw_pos_deg=row[f"{lidar_wtg}_yaw_position"])
        u = fmodel.sample_flow_at_points(points["x"].to_numpy(), points["y"].to_numpy(), points["z"].to_numpy())[0]
        rows[timestamp] = pd.Series(los_magnitude(u=u, wd_deg=row["wd"], points=points), index=points.index)
    return pd.concat(rows, axis=1).T.rename_axis(df.index.name)


def zxtm_minute_means(zx_df: pd.DataFrame) -> pd.DataFrame:
    """Return the minute-mean LOS speed per (range, segment height, side) from ZXTM fastlog rows.

    A probe-minute with fewer than ``MIN_SAMPLES_PER_MINUTE`` samples is NaN.
    """
    columns = {}
    for range_m in PROBE_RANGES_M:
        at_range = zx_df.loc[zx_df[RANGE_COL] == range_m]
        for height_m in SEGMENT_HEIGHTS_M:
            for side in SIDES:
                resampled = at_range[los_col(side, height_m)].resample(pd.Timedelta(minutes=1))
                columns[(range_m, height_m, side)] = resampled.mean().where(resampled.count() >= MIN_SAMPLES_PER_MINUTE)
    means = pd.DataFrame(columns)
    means.columns = means.columns.set_names(SEGMENT_LEVELS)
    return means


def load_zxtm_minute_df(*, start_dt: pd.Timestamp, end_dt_excl: pd.Timestamp) -> pd.DataFrame:
    """Load the ZXTM fastlog and return ``zxtm_minute_means`` for ``start_dt`` to ``end_dt_excl``."""
    zx_df = load_zx_lidar_fl_data(
        data_dir=get_data_dir() / "lidar_data",
        lidar_unit_id=ZXTM_UNIT_ID,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )
    if zx_df.empty:
        msg = f"no ZXTM {ZXTM_UNIT_ID} fastlog data from {start_dt} to {end_dt_excl}"
        raise ValueError(msg)
    return zxtm_minute_means(zx_df)


def left_minus_right(df: pd.DataFrame, *, height_m: float = HUB_HEIGHT_M) -> pd.DataFrame:
    """Return left minus right LOS at ``height_m``, one column per range."""

    def side(name: str) -> pd.DataFrame:
        return df.xs((height_m, name), axis=1, level=["segment_height_m", "side"])

    return side("Left") - side("Right")
