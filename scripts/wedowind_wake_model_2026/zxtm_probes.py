"""ZXTM nacelle LiDAR probes on T07 for the public FLORIS model of HOT.

Assumed ZXTM geometry and processing, to be confirmed by ZX:

- The ZXTM scans a cone of 15 deg half-angle around T07's rotor axis at about 50 Hz and post-processes
  the scan to the ~1 Hz records, one range at a time.
- The cone apex is at T07's hub: the tower position at 59 m. The nacelle offset and the lidar's height above
  the hub are ignored. The ZXTM corrects its measurements to zero inclination, so the cone axis is horizontal
  along T07's yaw position.
- ``Range (m)`` is the distance along the axis, so the cone circle at range R has radius R tan 15 deg.
- A rotor segment is a height band of D / 5 = 16.4 m centred on its listed height. The ``Left`` and ``Right``
  LOS speeds average the scan points in that band on the left and right halves of the cone, seen from the
  lidar looking upwind. A segment the cone does not reach has no data.
- The ZXTM is a continuous-wave lidar, so it measures the magnitude of the LOS speed.

The modelled LOS speed is FLORIS's streamwise speed, along the free-stream wind direction, projected onto
each beam; FLORIS's cross-flow from wake deflection is ignored. T07 itself is at zero yaw in FLORIS while the
cone follows T07's measured yaw position. Empirical Gauss has no induction zone, so the modelled 48 m and
64 m probes miss the slow-down in front of T07.

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
ROTOR_DIAMETER_M = 82.0
SEGMENT_HEIGHTS_M = (26.2, 42.6, 59.0, 75.4, 91.8)
SEGMENT_BAND_M = ROTOR_DIAMETER_M / 5
PROBE_RANGES_M = (48.0, 64.0, 85.0, 126.0, 208.0)
SIDES = ("Left", "Right")
N_SCAN_POINTS = 180
MIN_SAMPLES_PER_MINUTE = 5
RANGE_COL = "Range (m)"
SEGMENT_LEVELS = ["range_m", "segment_height_m", "side"]


def los_col(side: str, height_m: float) -> str:
    return f"{side} LOS Speed (m/s) at Rotor Segment Height {height_m:.1f}m"


def segment_height(z: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the centre height of the rotor segment band containing each height ``z``, NaN outside the rotor."""
    bottom = SEGMENT_HEIGHTS_M[0] - SEGMENT_BAND_M / 2
    band = np.floor((np.asarray(z, dtype=float) - bottom) / SEGMENT_BAND_M)
    inside = (band >= 0) & (band < len(SEGMENT_HEIGHTS_M))
    centres = np.array(SEGMENT_HEIGHTS_M)
    return np.where(inside, centres[np.clip(band, 0, len(centres) - 1).astype(int)], np.nan)


def cone_points(*, apex_x: float, apex_y: float, yaw_pos_deg: float) -> pd.DataFrame:
    """Return the scan points of every probe range in farm coordinates, with beam unit vectors and segments.

    ``yaw_pos_deg`` is the compass direction the rotor faces.
    """
    azimuth = np.tile(np.deg2rad((np.arange(N_SCAN_POINTS) + 0.5) * 360 / N_SCAN_POINTS), len(PROBE_RANGES_M))
    ranges = np.repeat(PROBE_RANGES_M, N_SCAN_POINTS)
    radius = ranges * np.tan(np.deg2rad(HALF_ANGLE_DEG))
    right = radius * np.sin(azimuth)
    up = radius * np.cos(azimuth)
    yaw = np.deg2rad(yaw_pos_deg)
    east = ranges * np.sin(yaw) + right * np.cos(yaw)
    north = ranges * np.cos(yaw) - right * np.sin(yaw)
    distance = ranges / np.cos(np.deg2rad(HALF_ANGLE_DEG))
    return pd.DataFrame(
        {
            "range_m": ranges,
            "side": np.where(right > 0, "Right", "Left"),
            "segment_height_m": segment_height(HUB_HEIGHT_M + up),
            "x": apex_x + east,
            "y": apex_y + north,
            "z": HUB_HEIGHT_M + up,
            "beam_x": east / distance,
            "beam_y": north / distance,
            "beam_z": up / distance,
        }
    )


def los_magnitude(*, u: npt.ArrayLike, wd_deg: float, points: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return |LOS| at ``points`` for a horizontal wind of speed ``u`` blowing from ``wd_deg``."""
    wd = np.deg2rad(wd_deg)
    along_beam = -np.sin(wd) * points["beam_x"].to_numpy() - np.cos(wd) * points["beam_y"].to_numpy()
    return np.abs(np.asarray(u, dtype=float) * along_beam)


def segment_means(points: pd.DataFrame, los: npt.ArrayLike) -> pd.Series:
    """Return the mean ``los`` per (range, segment height, side), as the ZXTM reports it."""
    return points.assign(los=los).dropna(subset=["segment_height_m"]).groupby(SEGMENT_LEVELS)["los"].mean()


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
    FLORIS runs one minute at a time because the cone moves with the lidar turbine's yaw.
    """
    yaw = np.zeros((len(df), len(turbine_names))) if yaw_angles is None else np.asarray(yaw_angles, dtype=float)
    lidar = turbine_names.index(lidar_wtg)
    apex_x, apex_y = float(fmodel.layout_x[lidar]), float(fmodel.layout_y[lidar])
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
        points = cone_points(apex_x=apex_x, apex_y=apex_y, yaw_pos_deg=row[f"{lidar_wtg}_yaw_position"])
        u = fmodel.sample_flow_at_points(points["x"].to_numpy(), points["y"].to_numpy(), points["z"].to_numpy())[0]
        rows[timestamp] = segment_means(points, los_magnitude(u=u, wd_deg=row["wd"], points=points))
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
