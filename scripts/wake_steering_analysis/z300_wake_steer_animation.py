r"""Animation of T11 (steering), T14 (downwind) and T13 (reference) plus the ZX300 vertical
profile LiDAR (unit 2428), located 279.68 m from T11 at bearing 10.66 deg.

Sibling of zxtm_wake_steer_animation.py — same overall layout (timeline on top, three
panels below) but the LiDAR panel becomes a vertical wind-speed profile, the centre
panel shows a single dot+arrow for the ground-based LiDAR, and a new right-hand panel
shows the wind-direction and vertical-wind-speed profiles on twin x-axes.

Outputs:
- z300_wake_steer_animation.gif
- frames/frame_*.png — still frames every 5 minutes of data time
- Static companion PNGs via zx300_wake_steers.plot_wake_steering_period_with_zx300
"""

import logging
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_fl_data, add_shear_and_veer
from hot_open.settings import get_filestore_dir, get_out_dir
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.inspect_data import (
    LOCAL_TEMPORARY_DIR,
    NORTH_CORRECTIONS,
    WTG_FL_ACT_POWER_COL,
    WTG_FL_YAW_POS_COL,
    ZX300_WD_COL,
    ZX300_WS_COL,
)
from scripts.wake_steering_analysis.zx300_wake_steers import (
    SMOOTHING_WINDOW as STATIC_PLOT_SMOOTHING_WINDOW,
)
from scripts.wake_steering_analysis.zx300_wake_steers import (
    plot_wake_steering_period_with_zx300,
)
from scripts.wake_steering_analysis.zxtm_wake_steer_animation import FixedPaletteWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
PLOT_START = pd.Timestamp("2026-04-04 03:30", tz="UTC")
PLOT_END = pd.Timestamp("2026-03-15 08:30", tz="UTC")
FRAME_INTERVAL_S = 30  # seconds of real data per animation frame
ANIMATION_DURATION_S = 60  # target GIF duration in seconds
SMOOTHING_WINDOW = 120  # rolling-mean window in seconds

# Load earlier so the rolling mean is fully populated at PLOT_START.
LOAD_START = PLOT_START - pd.Timedelta(seconds=SMOOTHING_WINDOW)

# ---------------------------------------------------------------------------
# Fixed geometry (metres)
# ---------------------------------------------------------------------------
ROTOR_DIAMETER_M = 82.0
NACELLE_FRONT_M = 6.0
NACELLE_REAR_M = 4.0
NACELLE_HALF_WIDTH_M = 2.0

ROTOR_LOWER_M = 18.0  # rotor extent shaded in the vertical-profile plots
ROTOR_UPPER_M = 100.0
LIDAR_BASE_OFFSET_M = 1.0  # ZX300 unit is 1 m tall: data at column "X m" is plotted at X+1 m
HUB_HEIGHT_PLOT_M = 58.0 + LIDAR_BASE_OFFSET_M

# ZX300 birds-eye location relative to T11
ZX300_DIST_M = 279.677632964692
ZX300_BEARING_DEG = 10.6646076181725

# Arrow length at ws_vmin and ws_vmax (interpolated linearly)
ARROW_LEN_MIN_M = 50.0
ARROW_LEN_MAX_M = 100.0

# ---------------------------------------------------------------------------
# ZX300 column patterns
# ---------------------------------------------------------------------------
HUB_WS_COL = ZX300_WS_COL  # "Horizontal Wind Speed (m/s) at 58m"
HUB_WD_COL = ZX300_WD_COL  # "Wind Direction (deg) at 58m"
SHEAR_COL = "Vertical Wind Shear Exponent"
_HORIZ_WS_PAT = re.compile(r"^Horizontal Wind Speed \(m/s\) at (\d+)m$")
_WD_PAT = re.compile(r"^Wind Direction \(deg\) at (\d+)m$")
_VERT_WS_PAT = re.compile(r"^Vertical Wind Speed \(m/s\) at (\d+)m$")

# ---------------------------------------------------------------------------
# Turbine config
# ---------------------------------------------------------------------------
WTG_NUMBERS = [11, 13, 14]
REF_NAME = "T13"
STEERING_NAME = "T11"
DEP_NAME = "T14"
TOGGLE_COL = "computed_driver_post_processed_toggle_state"
YAW_POS_COL = "computed_driver_pre_processed_yaw_direction_true_degrees"
WD_COL = "computed_core_post_processed_consensus_wind_direction_true_degrees"

# Coordinates from floris_config/hot_emgauss.yaml (0-indexed: T11=10, T13=12, T14=13)
TURBINE_COORDS: dict[str, tuple[float, float]] = {
    "T11": (498.8579110533185, 1734.8593213645222),
    "T13": (649.4328573605469, 2271.5147688163343),
    "T14": (841.688479019941, 1948.440198202186),
}

_t11_x, _t11_y = TURBINE_COORDS["T11"]
_zx300_bearing_rad = np.deg2rad(ZX300_BEARING_DEG)
ZX300_XY: tuple[float, float] = (
    _t11_x + ZX300_DIST_M * np.sin(_zx300_bearing_rad),
    _t11_y + ZX300_DIST_M * np.cos(_zx300_bearing_rad),
)


# ---------------------------------------------------------------------------
# Small helpers (mirrors zxtm_wake_steer_animation private helpers)
# ---------------------------------------------------------------------------


def _local_to_world(tx: float, ty: float, yaw_rad: float, dx: float, dy: float) -> tuple[float, float]:
    """Convert local (x_right, y_forward) offset to world (east, north) position."""
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    return tx + dx * cos_y + dy * sin_y, ty - dx * sin_y + dy * cos_y


def _shade_toggle(ax: plt.Axes, *, steer_df: pd.DataFrame, toggle_col: str) -> None:
    ylow, yhigh = ax.get_ylim()
    ax.fill_between(
        steer_df.index,
        ylow,
        yhigh,
        where=steer_df[toggle_col] == 0,
        color="red",
        alpha=0.05,
        label="steering off",
    )
    ax.fill_between(
        steer_df.index,
        ylow,
        yhigh,
        where=steer_df[toggle_col] == 1,
        color="green",
        alpha=0.05,
        label="steering on",
    )
    ax.set_ylim(ylow, yhigh)


def _add_smoothed(df: pd.DataFrame, *, col: str, new_col: str, min_periods: int = SMOOTHING_WINDOW // 2) -> None:
    df[new_col] = df[col].rolling(window=SMOOTHING_WINDOW, min_periods=min_periods).mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_wtg_data() -> dict[str, pd.DataFrame]:
    wtg_fl_df = load_hot_fl_data(
        data_dir=get_filestore_dir(),
        wtg_numbers=WTG_NUMBERS,
        start_dt=LOAD_START,
        # +1 s so PLOT_END is itself included after the inclusive .loc clip below.
        end_dt_excl=PLOT_END + pd.Timedelta(seconds=1),
        extra_tags=[TOGGLE_COL, YAW_POS_COL, WD_COL],
    )
    for wtg, correction in NORTH_CORRECTIONS.items():
        if wtg in wtg_fl_df.columns.get_level_values(0):
            wtg_fl_df[(wtg, WTG_FL_YAW_POS_COL)] = (wtg_fl_df[(wtg, WTG_FL_YAW_POS_COL)] + correction) % 360

    result: dict[str, pd.DataFrame] = {}
    for name in (REF_NAME, STEERING_NAME, DEP_NAME):
        df = wtg_fl_df[name].copy()
        _add_smoothed(df, col=WTG_FL_ACT_POWER_COL, new_col="smoothed_pw")
        result[name] = df.loc[PLOT_START:PLOT_END]
    return result


def _detect_heights(df: pd.DataFrame) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Discover ZX300 profile columns and return {height_m: column} for each variable."""
    horiz_ws: dict[int, str] = {}
    wd: dict[int, str] = {}
    vert_ws: dict[int, str] = {}
    for col in df.columns:
        m = _HORIZ_WS_PAT.match(col)
        if m:
            horiz_ws[int(m.group(1))] = col
            continue
        m = _WD_PAT.match(col)
        if m:
            wd[int(m.group(1))] = col
            continue
        m = _VERT_WS_PAT.match(col)
        if m:
            vert_ws[int(m.group(1))] = col
    return horiz_ws, wd, vert_ws


def _load_zx300_data() -> tuple[pd.DataFrame, dict[int, str], dict[int, str], dict[int, str]]:
    """Load ZX300 fastlog, detect profile heights, smooth across all profile + hub columns."""
    zx300_fl_df = load_zx_lidar_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="2428",
        start_dt=LOAD_START,
        end_dt_excl=PLOT_END + pd.Timedelta(seconds=1),
        remove_bad_values=True,
    )
    zx300_fl_df = add_shear_and_veer(zx300_fl_df)
    horiz_ws, wd, vert_ws = _detect_heights(zx300_fl_df)

    smoothed_cols = sorted({*horiz_ws.values(), *wd.values(), *vert_ws.values()})
    # Time-window rolling so the smoothing length is "120 seconds of data" regardless of
    # sampling rate (ZX300 fastlog is not necessarily 1 Hz).
    smoothed = zx300_fl_df[smoothed_cols].sort_index().rolling(window=f"{SMOOTHING_WINDOW}s", min_periods=1).mean()
    smoothed = smoothed.loc[PLOT_START:PLOT_END]
    return smoothed, horiz_ws, wd, vert_ws


def _compute_color_bounds(
    wtg_data: dict[str, pd.DataFrame],
    zx300_smoothed: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Return (pw_vmin, pw_vmax, ws_vmin, ws_vmax). WS bounds are hub-height only — per
    the design, only the hub-height dot/arrow are colour-mapped; profile lines stay plain.
    """
    all_pw = pd.concat([df["smoothed_pw"].dropna() for df in wtg_data.values()])
    pw_vmin = float(all_pw.min())
    pw_vmax = float(all_pw.max())

    hub_ws = zx300_smoothed[HUB_WS_COL].dropna()
    if hub_ws.empty:
        ws_vmin, ws_vmax = 0.0, 15.0
    else:
        ws_vmin = float(hub_ws.min())
        ws_vmax = float(hub_ws.max())
    return pw_vmin, pw_vmax, ws_vmin, ws_vmax


# ---------------------------------------------------------------------------
# Figure construction
# ---------------------------------------------------------------------------


def _init_figure(
    *,
    wtg_data: dict[str, pd.DataFrame],
    zx300_smoothed: pd.DataFrame,
    horiz_ws_cols: dict[int, str],
    wd_cols: dict[int, str],
    vert_ws_cols: dict[int, str],
    pw_vmin: float,
    pw_vmax: float,
    ws_vmin: float,
    ws_vmax: float,
) -> dict:
    """Build the figure once; per-frame artists are placeholders updated in the loop."""
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 3],
        hspace=0.32,
        wspace=0.40,
        left=0.06,
        right=0.84,
        top=0.92,
        bottom=0.12,
    )
    ax_tl = fig.add_subplot(gs[0, :])
    ax_ws = fig.add_subplot(gs[1, 0])  # bottom-left: horizontal WS vertical profile
    ax_be = fig.add_subplot(gs[1, 1])  # bottom-centre: birds-eye
    ax_wd = fig.add_subplot(gs[1, 2])  # bottom-right: WD profile + (twin) Vert WS profile
    ax_wd_top = ax_wd.twiny()

    fig.suptitle(
        f"Hill of Towie — T11 wake steering for T14 (T13 reference) — {PLOT_START:%Y-%m-%d %H:%M}–{PLOT_END:%H:%M} UTC",
        fontsize=14,
    )

    # --- Timeline --------------------------------------------------------------
    ref_df = wtg_data[REF_NAME]
    steer_df = wtg_data[STEERING_NAME]
    dep_df = wtg_data[DEP_NAME]
    for c_idx, (name, df) in enumerate([(REF_NAME, ref_df), (STEERING_NAME, steer_df), (DEP_NAME, dep_df)]):
        c = f"C{c_idx}"
        ax_tl.plot(df.index, df[WD_COL], color=c, linestyle=":", alpha=0.5, label=f"{name} wind dir")
        ax_tl.plot(df.index, df[YAW_POS_COL], color=c, label=f"{name} yaw pos")
    ax_tl.plot(
        zx300_smoothed.index,
        zx300_smoothed[HUB_WD_COL],
        color="C3",
        linestyle=":",
        alpha=0.8,
        label="ZX300 wd@HH",
    )
    _shade_toggle(ax_tl, steer_df=steer_df, toggle_col=TOGGLE_COL)
    ax_tl.set_xlim(PLOT_START, PLOT_END)
    ax_tl.set_ylabel("direction [degN]")
    ax_tl.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, frameon=True)
    ax_tl.grid(True, alpha=0.4)
    plt.setp(ax_tl.xaxis.get_majorticklabels(), rotation=45, ha="right")
    vline = ax_tl.axvline(PLOT_START, color="black", linewidth=1.5, linestyle="--", zorder=5)

    # --- Colour maps + norms (shared across the three lower panels) -----------
    pw_norm = Normalize(pw_vmin, pw_vmax)
    pw_cmap = plt.cm.plasma
    ws_norm = Normalize(ws_vmin, ws_vmax)
    ws_cmap = pw_cmap  # one colour story per quantity; matches the zxtm sibling

    # --- Bottom-left: horizontal WS vertical profile ---------------------------
    horiz_heights = sorted(horiz_ws_cols)
    horiz_plot_heights = [h + LIDAR_BASE_OFFSET_M for h in horiz_heights]
    horiz_data_all = zx300_smoothed[[horiz_ws_cols[h] for h in horiz_heights]].to_numpy()
    horiz_finite = horiz_data_all[np.isfinite(horiz_data_all)]
    if horiz_finite.size:
        ws_x_lo = float(horiz_finite.min()) - 0.5
        ws_x_hi = float(horiz_finite.max()) + 0.5
    else:
        ws_x_lo, ws_x_hi = 0.0, 20.0
    profile_y_top = max(horiz_plot_heights) + 5
    ax_ws.set_xlim(ws_x_lo, ws_x_hi)
    ax_ws.set_ylim(0, profile_y_top)
    ax_ws.set_xlabel("horizontal wind speed (m/s)", fontsize=10)
    ax_ws.set_ylabel("height above ground (m)", fontsize=10)
    ax_ws.set_title("ZX300 wind speed profile", fontsize=10)
    ax_ws.grid(True, alpha=0.4)
    ax_ws.axhspan(ROTOR_LOWER_M, ROTOR_UPPER_M, color="gray", alpha=0.15, label="rotor extent")
    (horiz_ws_line,) = ax_ws.plot(
        [np.nan] * len(horiz_plot_heights),
        horiz_plot_heights,
        color="C0",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="horizontal WS",
    )
    hub_ws_dot = ax_ws.scatter(
        [ws_vmin],
        [HUB_HEIGHT_PLOT_M],
        c=[(ws_vmin + ws_vmax) / 2],
        s=[80.0],
        cmap=ws_cmap,
        norm=ws_norm,
        zorder=7,
        edgecolors="k",
        linewidths=0.5,
        label="hub-height WS",
    )

    hub_ws_label = ax_ws.annotate(
        None,
        xy=(ws_vmin, HUB_HEIGHT_PLOT_M),  # point to annotate
        xytext=(5, 0),  # offset (points)
        textcoords="offset points",
        ha="left",  # align text to the left (so it appears right of point)
        va="center",
        fontsize = 10
    )

    ax_ws.legend(loc="upper left", fontsize=8)

    # --- Bottom-centre: birds-eye ---------------------------------------------
    all_x = [c[0] for c in TURBINE_COORDS.values()] + [ZX300_XY[0]]
    all_y = [c[1] for c in TURBINE_COORDS.values()] + [ZX300_XY[1]]
    ax_be.set_xlim(min(all_x) - 100, max(all_x) + 100)
    ax_be.set_ylim(min(all_y) - 100, max(all_y) + 110)
    ax_be.set_aspect("equal", adjustable="datalim")
    ax_be.set_xlabel("easting (m)", fontsize=10)
    ax_be.set_ylabel("northing (m)", fontsize=10)
    ax_be.grid(True, alpha=0.3)
    title_artist = ax_be.set_title("", fontsize=10)

    for name, (tx, ty) in TURBINE_COORDS.items():
        ax_be.add_patch(mpatches.Circle((tx, ty), radius=1.5, facecolor="gray", edgecolor="black", zorder=4))
        ax_be.text(tx, ty + 50, name, ha="center", va="bottom", fontsize=10, fontweight="bold", zorder=6)

    turbine_artists: dict[str, tuple[mpatches.Polygon, Line2D, plt.Text]] = {}
    for name, (tx, ty) in TURBINE_COORDS.items():
        nacelle = mpatches.Polygon(
            [[tx, ty]] * 4,
            closed=True,
            facecolor="lightgray",
            edgecolor="black",
            zorder=3,
        )
        ax_be.add_patch(nacelle)
        (rotor,) = ax_be.plot(
            [tx, tx],
            [ty, ty],
            color=pw_cmap(0.5),
            linewidth=5,
            solid_capstyle="round",
            zorder=5,
        )
        power_text = ax_be.text(tx, ty + 35, "", ha="center", va="bottom", fontsize=10, zorder=6)
        turbine_artists[name] = (nacelle, rotor, power_text)

    # ZX300 birds-eye marker: dot + arrow + label + ws-value text.
    zx_x, zx_y = ZX300_XY
    zx300_scatter = ax_be.scatter(
        [zx_x],
        [zx_y],
        c=[(ws_vmin + ws_vmax) / 2],
        s=[80.0],
        cmap=ws_cmap,
        norm=ws_norm,
        zorder=6,
        edgecolors="k",
        linewidths=0.5,
    )
    zx300_arrow = FancyArrowPatch(
        (zx_x, zx_y),
        (zx_x, zx_y),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=2.5,
        color=ws_cmap(0.5),
        zorder=5,
    )
    ax_be.add_patch(zx300_arrow)
    ax_be.text(
        zx_x, zx_y + 50, "ZX300", ha="center", va="bottom", fontsize=10, fontweight="bold", color="navy", zorder=6
    )
    zx300_ws_text = ax_be.text(zx_x, zx_y + 35, "", ha="center", va="bottom", fontsize=10, zorder=6)

    # --- Bottom-right: WD + Vert WS vertical profiles --------------------------
    wd_heights = sorted(wd_cols)
    wd_plot_heights = [h + LIDAR_BASE_OFFSET_M for h in wd_heights]
    vert_heights = sorted(vert_ws_cols)
    vert_plot_heights = [h + LIDAR_BASE_OFFSET_M for h in vert_heights]

    vert_data_all = zx300_smoothed[[vert_ws_cols[h] for h in vert_heights]].to_numpy()
    vert_finite = vert_data_all[np.isfinite(vert_data_all)]
    if vert_finite.size:
        vert_x_lo = float(vert_finite.min()) - 0.1
        vert_x_hi = float(vert_finite.max()) + 0.1
    else:
        vert_x_lo, vert_x_hi = -1.0, 1.0

    wd_data_all = zx300_smoothed[[wd_cols[h] for h in wd_heights]].to_numpy()
    wd_finite = wd_data_all[np.isfinite(wd_data_all)]
    if wd_finite.size:
        wd_x_lo = float(wd_finite.min()) - 2.0
        wd_x_hi = float(wd_finite.max()) + 2.0
    else:
        wd_x_lo, wd_x_hi = 0.0, 360.0

    wd_y_top = max([*wd_plot_heights, *vert_plot_heights]) + 5
    ax_wd.set_xlim(wd_x_lo, wd_x_hi)
    ax_wd.set_ylim(0, wd_y_top)
    ax_wd.set_xlabel("wind direction (degN)", fontsize=10, color="C0")
    ax_wd.tick_params(axis="x", labelcolor="C0")
    ax_wd.set_ylabel("height above ground (m)", fontsize=10)
    ax_wd.set_title("ZX300 wind direction & vertical WS", fontsize=10)
    ax_wd.grid(True, alpha=0.4)
    ax_wd.axhspan(ROTOR_LOWER_M, ROTOR_UPPER_M, color="gray", alpha=0.15)
    ax_wd_top.set_xlim(vert_x_lo, vert_x_hi)
    ax_wd_top.set_xlabel("vertical wind speed (m/s)", fontsize=10, color="C3")
    ax_wd_top.tick_params(axis="x", labelcolor="C3")

    (wd_profile_line,) = ax_wd.plot(
        [np.nan] * len(wd_plot_heights),
        wd_plot_heights,
        color="C0",
        marker="o",
        markersize=4,
        linewidth=1.5,
        label="wind direction",
    )
    (vert_profile_line,) = ax_wd_top.plot(
        [np.nan] * len(vert_plot_heights),
        vert_plot_heights,
        color="C3",
        marker="s",
        markersize=4,
        linewidth=1.5,
        label="vertical WS",
    )
    legend_lines = [wd_profile_line, vert_profile_line]
    ax_wd.legend(legend_lines, [ln.get_label() for ln in legend_lines], loc="upper right", fontsize=8)

    # --- Colourbars (positioned in figure coords, same as zxtm sibling) -------
    cbar_y = 0.04
    cbar_h = 0.013
    cb_left = 0.10
    cb_right = 0.78
    cb_gap = 0.06
    cb_width = (cb_right - cb_left - cb_gap) / 2

    cax_pw = fig.add_axes([cb_left, cbar_y, cb_width, cbar_h])
    pw_sm = plt.cm.ScalarMappable(norm=pw_norm, cmap=pw_cmap)
    pw_sm.set_array([])
    cb_pw = fig.colorbar(pw_sm, cax=cax_pw, orientation="horizontal")
    cb_pw.set_label("power (MW)", fontsize=10)
    cb_pw.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.1f}"))
    cb_pw.ax.tick_params(labelsize=7)

    cax_ws = fig.add_axes([cb_left + cb_width + cb_gap, cbar_y, cb_width, cbar_h])
    ws_sm = plt.cm.ScalarMappable(norm=ws_norm, cmap=ws_cmap)
    ws_sm.set_array([])
    cb_ws = fig.colorbar(ws_sm, cax=cax_ws, orientation="horizontal")
    cb_ws.set_label("ZX300 hub-height wind speed (m/s)", fontsize=10)
    cb_ws.ax.tick_params(labelsize=7)

    return {
        "fig": fig,
        "vline": vline,
        "title_artist": title_artist,
        "turbine_artists": turbine_artists,
        "zx300_scatter": zx300_scatter,
        "zx300_arrow": zx300_arrow,
        "zx300_ws_text": zx300_ws_text,
        "horiz_ws_line": horiz_ws_line,
        "horiz_heights": horiz_heights,
        "horiz_ws_cols": horiz_ws_cols,
        "hub_ws_dot": hub_ws_dot,
        "hub_ws_label": hub_ws_label,
        "wd_profile_line": wd_profile_line,
        "wd_heights": wd_heights,
        "wd_cols": wd_cols,
        "vert_profile_line": vert_profile_line,
        "vert_heights": vert_heights,
        "vert_ws_cols": vert_ws_cols,
        "pw_norm": pw_norm,
        "pw_cmap": pw_cmap,
        "ws_norm": ws_norm,
        "ws_cmap": ws_cmap,
    }


# ---------------------------------------------------------------------------
# Animation loop
# ---------------------------------------------------------------------------


def _run_animation(
    *,
    wtg_data: dict[str, pd.DataFrame],
    zx300_smoothed: pd.DataFrame,
    horiz_ws_cols: dict[int, str],
    wd_cols: dict[int, str],
    vert_ws_cols: dict[int, str],
    pw_vmin: float,
    pw_vmax: float,
    ws_vmin: float,
    ws_vmax: float,
    out_dir: Path,
    smoke_test: bool = False,
) -> None:
    handles = _init_figure(
        wtg_data=wtg_data,
        zx300_smoothed=zx300_smoothed,
        horiz_ws_cols=horiz_ws_cols,
        wd_cols=wd_cols,
        vert_ws_cols=vert_ws_cols,
        pw_vmin=pw_vmin,
        pw_vmax=pw_vmax,
        ws_vmin=ws_vmin,
        ws_vmax=ws_vmax,
    )
    fig = handles["fig"]
    vline = handles["vline"]
    title_artist = handles["title_artist"]
    turbine_artists = handles["turbine_artists"]
    zx300_scatter = handles["zx300_scatter"]
    zx300_arrow = handles["zx300_arrow"]
    zx300_ws_text = handles["zx300_ws_text"]
    horiz_ws_line = handles["horiz_ws_line"]
    horiz_heights = handles["horiz_heights"]
    hub_ws_dot = handles["hub_ws_dot"]
    hub_ws_label = handles["hub_ws_label"]
    wd_profile_line = handles["wd_profile_line"]
    wd_heights = handles["wd_heights"]
    vert_profile_line = handles["vert_profile_line"]
    vert_heights = handles["vert_heights"]
    pw_norm = handles["pw_norm"]
    pw_cmap = handles["pw_cmap"]
    ws_norm = handles["ws_norm"]
    ws_cmap = handles["ws_cmap"]

    ref_df = wtg_data[REF_NAME]
    ws_fill = (ws_vmin + ws_vmax) / 2

    timestamps = ref_df.resample(f"{FRAME_INTERVAL_S}s").first().index
    fps = len(timestamps) / ANIMATION_DURATION_S

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "z300_wake_steer_animation.gif"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    logger.info("rendering %d frames at %.1f fps -> %s", len(timestamps), fps, output_path)
    logger.info("also saving 5-min still frames to %s", frames_dir)

    writer = FixedPaletteWriter(fps=fps)
    if smoke_test:
        timestamps = timestamps[:50]
    pngs_saved = 0

    nacelle_corners_local = np.array(
        [
            [-NACELLE_HALF_WIDTH_M, -NACELLE_REAR_M],
            [NACELLE_HALF_WIDTH_M, -NACELLE_REAR_M],
            [NACELLE_HALF_WIDTH_M, NACELLE_FRONT_M],
            [-NACELLE_HALF_WIDTH_M, NACELLE_FRONT_M],
        ]
    )
    half_r = ROTOR_DIAMETER_M / 2
    zx_x, zx_y = ZX300_XY
    ws_range = max(ws_vmax - ws_vmin, 1e-6)

    horiz_col_list = [horiz_ws_cols[h] for h in horiz_heights]
    wd_col_list = [wd_cols[h] for h in wd_heights]
    vert_col_list = [vert_ws_cols[h] for h in vert_heights]

    with writer.saving(fig, str(output_path), dpi=100):
        for i, ts in enumerate(tqdm(timestamps, desc="frames")):
            title_artist.set_text(ts.strftime("%H:%M:%S UTC"))

            # --- Turbines -----------------------------------------------------
            for name, (tx, ty) in TURBINE_COORDS.items():
                df = wtg_data[name]
                idx = df.index.get_indexer([ts], method="nearest")[0]
                row = df.iloc[idx]
                raw_yaw = row[YAW_POS_COL]
                yaw = float(raw_yaw) if not pd.isna(raw_yaw) else 0.0
                power = float(row["smoothed_pw"]) if not pd.isna(row["smoothed_pw"]) else 0.0

                nacelle, rotor, power_text = turbine_artists[name]
                yaw_rad = np.deg2rad(yaw)
                cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
                dx = nacelle_corners_local[:, 0]
                dy = nacelle_corners_local[:, 1]
                world_x = tx + dx * cos_y + dy * sin_y
                world_y = ty - dx * sin_y + dy * cos_y
                nacelle.set_xy(np.column_stack([world_x, world_y]))

                r_left = _local_to_world(tx, ty, yaw_rad, -half_r, NACELLE_FRONT_M)
                r_right = _local_to_world(tx, ty, yaw_rad, half_r, NACELLE_FRONT_M)
                rotor.set_data([r_left[0], r_right[0]], [r_left[1], r_right[1]])
                rotor.set_color(pw_cmap(pw_norm(power)))

                power_text.set_text(f"{power / 1000:.1f} MW")

            # --- ZX300 hub-height dot + arrow + label -------------------------
            zx_idx = zx300_smoothed.index.get_indexer([ts], method="nearest")[0]
            zx_row = zx300_smoothed.iloc[zx_idx] if zx_idx >= 0 else None
            hub_ws = zx_row[HUB_WS_COL] if zx_row is not None else np.nan
            hub_wd = zx_row[HUB_WD_COL] if zx_row is not None else np.nan
            shear = zx_row[SHEAR_COL] if zx_row is not None else np.nan

            if pd.isna(hub_ws):
                zx300_scatter.set_array(np.array([ws_fill]))
                zx300_scatter.set_sizes([40.0])
            else:
                zx300_scatter.set_array(np.array([float(hub_ws)]))
                zx300_scatter.set_sizes([40.0 + float(ws_norm(hub_ws)) * 100.0])

            if pd.isna(hub_ws) or pd.isna(hub_wd):
                zx300_arrow.set_positions((zx_x, zx_y), (zx_x, zx_y))
            else:
                # Met wind direction = direction wind comes FROM. Arrow points downwind
                # (where wind is going), so wd=270° -> arrow east, wd=0° -> arrow south.
                going_dx = -np.sin(np.deg2rad(hub_wd))
                going_dy = -np.cos(np.deg2rad(hub_wd))
                t = float(np.clip((float(hub_ws) - ws_vmin) / ws_range, 0.0, 1.0))
                length = ARROW_LEN_MIN_M + t * (ARROW_LEN_MAX_M - ARROW_LEN_MIN_M)
                tip = (zx_x + going_dx * length, zx_y + going_dy * length)
                zx300_arrow.set_positions((zx_x, zx_y), tip)
                zx300_arrow.set_color(ws_cmap(ws_norm(hub_ws)))

            zx300_ws_text.set_text("" if pd.isna(hub_ws) else f"{float(hub_ws):.1f} m/s")

            # --- Bottom-left: horizontal WS profile + hub-height coloured dot -
            if zx_row is not None:
                horiz_vals = [zx_row.get(col, np.nan) for col in horiz_col_list]
            else:
                horiz_vals = [np.nan] * len(horiz_col_list)
            horiz_ws_line.set_xdata(horiz_vals)
            if pd.isna(hub_ws):
                hub_ws_dot.set_offsets(np.empty((0, 2)))
                hub_ws_label.set_visible(False)

            else:
                hub_ws_dot.set_offsets([[float(hub_ws), HUB_HEIGHT_PLOT_M]])
                hub_ws_label.position([float(hub_ws), HUB_HEIGHT_PLOT_M])
                hub_ws_label.set_text(f"α = {shear:.3f}")
                hub_ws_label.set_visible(True)
                hub_ws_dot.set_array(np.array([float(hub_ws)]))
                hub_ws_dot.set_sizes([80.0])

            # --- Bottom-right: WD + vertical WS profiles ----------------------
            if zx_row is not None:
                wd_vals = [zx_row.get(col, np.nan) for col in wd_col_list]
                vert_vals = [zx_row.get(col, np.nan) for col in vert_col_list]
            else:
                wd_vals = [np.nan] * len(wd_col_list)
                vert_vals = [np.nan] * len(vert_col_list)
            wd_profile_line.set_xdata(wd_vals)
            vert_profile_line.set_xdata(vert_vals)

            vline.set_xdata([ts, ts])

            # Dump still PNGs every 5 minutes of data time.
            if ts.minute % 5 == 0 and ts.second == 0:
                png_path = frames_dir / f"frame_{ts:%Y%m%d_%H%M}.png"
                fig.savefig(png_path, dpi=100)
                pngs_saved += 1

            writer.grab_frame()
            if i % 200 == 0:
                logger.info("frame %d / %d", i, len(timestamps))

    logger.info("saved %s", output_path)
    logger.info("saved %d still PNGs to %s", pngs_saved, frames_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Static companion plots — same style as zx300_wake_steers.py
# ---------------------------------------------------------------------------


def _save_static_pngs(out_dir: Path) -> None:
    load_start = PLOT_START - pd.Timedelta(seconds=STATIC_PLOT_SMOOTHING_WINDOW)

    wtg_fl_df = load_hot_fl_data(
        data_dir=get_filestore_dir(),
        wtg_numbers=WTG_NUMBERS,
        start_dt=load_start,
        end_dt_excl=PLOT_END,
        extra_tags=[TOGGLE_COL, YAW_POS_COL, WD_COL],
    )
    for wtg, correction in NORTH_CORRECTIONS.items():
        if wtg in wtg_fl_df.columns.get_level_values(0):
            wtg_fl_df[(wtg, WTG_FL_YAW_POS_COL)] = (wtg_fl_df[(wtg, WTG_FL_YAW_POS_COL)] + correction) % 360

    smoothed_pw_col = "smoothed_pw"
    dfs: dict[str, pd.DataFrame] = {}
    for name in (REF_NAME, STEERING_NAME, DEP_NAME):
        df = wtg_fl_df[name].copy()
        df[smoothed_pw_col] = (
            df[WTG_FL_ACT_POWER_COL]
            .rolling(window=STATIC_PLOT_SMOOTHING_WINDOW, min_periods=STATIC_PLOT_SMOOTHING_WINDOW // 2)
            .mean()
        )
        dfs[name] = df[(df.index >= PLOT_START) & (df.index < PLOT_END)]

    zx300_fl_df = load_zx_lidar_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="2428",
        start_dt=load_start,
        end_dt_excl=PLOT_END,
        remove_bad_values=True,
    )
    zx300_fl_df = zx300_fl_df[(zx300_fl_df.index >= PLOT_START) & (zx300_fl_df.index < PLOT_END)]

    plot_wake_steering_period_with_zx300(
        plot_ref_df=dfs[REF_NAME],
        plot_steer_df=dfs[STEERING_NAME],
        plot_dep_df=dfs[DEP_NAME],
        plot_zx300_df=zx300_fl_df,
        ref_name=REF_NAME,
        steering_name=STEERING_NAME,
        dependent_turbine_name=DEP_NAME,
        wd_col=WD_COL,
        yawpos_col=YAW_POS_COL,
        toggle_col=TOGGLE_COL,
        smoothed_pw_col=smoothed_pw_col,
        plot_start=PLOT_START,
        plot_end=PLOT_END,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    smoke_test = False
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    logger.info("log file is at %s", log_path)

    logger.info("saving static companion PNGs (zx300_wake_steers.py style)...")
    _save_static_pngs(out_dir)

    logger.info("loading WTG fastlog data...")
    wtg_data = _load_wtg_data()

    logger.info("loading ZX300 lidar data...")
    zx300_smoothed, horiz_ws_cols, wd_cols, vert_ws_cols = _load_zx300_data()
    logger.info(
        "ZX300 detected heights — horizontal WS: %d, wind direction: %d, vertical WS: %d",
        len(horiz_ws_cols),
        len(wd_cols),
        len(vert_ws_cols),
    )

    logger.info("ZX300 birds-eye location: easting=%.2f m, northing=%.2f m", *ZX300_XY)

    logger.info("computing colour bounds...")
    pw_vmin, pw_vmax, ws_vmin, ws_vmax = _compute_color_bounds(wtg_data, zx300_smoothed)
    logger.info(
        "power %.0f–%.0f kW  |  hub wind speed %.1f–%.1f m/s",
        pw_vmin,
        pw_vmax,
        ws_vmin,
        ws_vmax,
    )

    _run_animation(
        wtg_data=wtg_data,
        zx300_smoothed=zx300_smoothed,
        horiz_ws_cols=horiz_ws_cols,
        wd_cols=wd_cols,
        vert_ws_cols=vert_ws_cols,
        pw_vmin=pw_vmin,
        pw_vmax=pw_vmax,
        ws_vmin=ws_vmin,
        ws_vmax=ws_vmax,
        out_dir=out_dir,
        smoke_test=smoke_test,
    )
