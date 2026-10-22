"""FLORIS against the ZXTM LiDAR on T07: modelled and measured LOS speeds on toggle-off and toggle-on minutes.

The model, free stream, ``k`` and T03 yaw are as in ``toggle_on_t07.py``; see ``hot_floris.py``. FLORIS is
sampled on the ZXTM scan cone, which follows T07's measured yaw position, and projected onto the beams; the
assumed ZXTM geometry is in ``zxtm_probes.py``. FLORIS also runs with zero yaw to show the modelled steering
effect. Profiles and metrics only use probe-minutes with a measurement.

Run with ``uv sync --group wedowind`` first.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from floris import FlorisModel, TimeSeries
from floris.flow_visualization import visualize_cut_plane
from floris.layout_visualization import plot_turbine_rotors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from hot_open.settings import get_out_dir
from hot_open.unpack import unpack_local_meta_data
from scripts.logger import setup_logger
from scripts.wedowind_wake_model_2026.hot_floris import (
    FREE_STREAM_COLS,
    PREDICTION_WTG,
    STEERING_WTG,
    WINDOW_END_EXCL,
    WINDOW_START,
    filter_minutes,
    fit_toggle_off_k,
    load_hot_floris_model,
    load_minute_df,
    shade_toggle_on,
    yaw_angles_for,
)
from scripts.wedowind_wake_model_2026.zxtm_probes import (
    HALF_ANGLE_DEG,
    HUB_HEIGHT_M,
    PROBE_RANGES_M,
    SIDES,
    cone_points,
    left_minus_right,
    load_zxtm_minute_df,
    modelled_los_minutes,
)

logger = logging.getLogger(__name__)

LIDAR_YAW_COL = f"{PREDICTION_WTG}_yaw_position"
MISALIGNMENT_COL = f"{STEERING_WTG}_yaw_misalignment"
MEASURED = "measured"
YAWED = f"FLORIS, {STEERING_WTG} measured yaw"
UNYAWED = "FLORIS, zero yaw"
SOURCE_STYLES = {MEASURED: "-o", YAWED: "--", UNYAWED: ":"}
SIDE_COLORS = {"Left": "tab:blue", "Right": "tab:orange"}
SNAPSHOT_MARGIN_M = 250.0
SNAPSHOT_RESOLUTION = 600


def los_error_metrics(*, measured: pd.Series, modelled: pd.Series) -> dict[str, float | int | None]:
    err = (modelled - measured).dropna()
    if err.empty:
        return {"rmse_m_s": None, "bias_m_s": None, "count": 0}
    return {"rmse_m_s": float(np.sqrt((err**2).mean())), "bias_m_s": float(err.mean()), "count": len(err)}


def plot_profiles(*, sources: dict[str, pd.DataFrame], index: pd.Index, toggle: str, k: float, out_dir: Path) -> None:
    """Plot mean |LOS| against height per range on ``index`` minutes for each source."""
    fig, axes = plt.subplots(1, len(PROBE_RANGES_M), figsize=(20, 6), sharey=True)
    for ax, range_m in zip(axes, PROBE_RANGES_M, strict=True):
        for side in SIDES:
            for label, df in sources.items():
                profile = df.loc[index].xs((range_m, side), axis=1, level=["range_m", "side"]).mean()
                ax.plot(
                    profile.to_numpy(),
                    profile.index,
                    SOURCE_STYLES[label],
                    color=SIDE_COLORS[side],
                    label=f"{side}, {label}",
                )
        ax.set_title(f"{range_m:g} m")
        ax.set_xlabel("mean |LOS| [m/s]")
        ax.grid(visible=True, alpha=0.3)
    axes[0].set_ylabel("height [m]")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    fig.suptitle(f"ZXTM on {PREDICTION_WTG}, {len(index)} toggle-{toggle} minutes, k={k:.3f}")
    plot_path = out_dir / f"profiles_toggle_{toggle}.png"
    logger.info("Writing: %s", plot_path)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hub_height_time_series(*, minute_df: pd.DataFrame, sources: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Plot left minus right |LOS| at hub height per range over the window."""
    differences = {label: left_minus_right(df) for label, df in sources.items()}
    fig, axes = plt.subplots(len(PROBE_RANGES_M), 1, figsize=(14, 3 * len(PROBE_RANGES_M)), sharex=True)
    for ax, range_m in zip(axes, PROBE_RANGES_M, strict=True):
        shade_toggle_on(ax, minute_df=minute_df)
        for label, diff in differences.items():
            ax.plot(diff.index, diff[range_m], drawstyle="steps-post", label=label)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{range_m:g} m left - right [m/s]")
        ax.grid(visible=True, alpha=0.3)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    axes[-1].set_xlabel("timestamp (UTC)")
    fig.suptitle(f"ZXTM on {PREDICTION_WTG}: left minus right |LOS| at {HUB_HEIGHT_M:g} m")
    plot_path = out_dir / "time_series_hub_height.png"
    logger.info("Writing: %s", plot_path)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_snapshot_minutes(
    *, plot_df: pd.DataFrame, measured: pd.DataFrame, on_index: pd.Index, off_index: pd.Index
) -> dict[str, pd.Timestamp]:
    """Return the toggle-on minute with the largest |T03 misalignment| and the toggle-off minute closest to it in
    wind direction, both with every hub-height probe measured.
    """
    hub_measured = measured.xs(HUB_HEIGHT_M, axis=1, level="segment_height_m").notna().all(axis=1)
    on_ok = on_index[hub_measured.loc[on_index].to_numpy()]
    off_ok = off_index[hub_measured.loc[off_index].to_numpy()]
    on_minute = plot_df.loc[on_ok, MISALIGNMENT_COL].abs().idxmax()
    wd_diff = (plot_df.loc[off_ok, "wd"] - plot_df.loc[on_minute, "wd"] + 180) % 360 - 180
    return {"toggle off": wd_diff.abs().idxmin(), "toggle on": on_minute}


def plot_flow_snapshots(
    *,
    fmodel: FlorisModel,
    turbine_names: list[str],
    plot_df: pd.DataFrame,
    measured: pd.DataFrame,
    minutes: dict[str, pd.Timestamp],
    k: float,
    out_dir: Path,
) -> None:
    """Plot the FLORIS hub-height plane per minute with the hub-height probes coloured by measured |LOS| / cos 15."""
    near = [turbine_names.index(wtg) for wtg in (STEERING_WTG, PREDICTION_WTG)]
    x_bounds = (fmodel.layout_x[near].min() - SNAPSHOT_MARGIN_M, fmodel.layout_x[near].max() + SNAPSHOT_MARGIN_M)
    y_bounds = (fmodel.layout_y[near].min() - SNAPSHOT_MARGIN_M, fmodel.layout_y[near].max() + SNAPSHOT_MARGIN_M)
    lidar = turbine_names.index(PREDICTION_WTG)
    norm = Normalize(vmin=0.0, vmax=1.05 * k * plot_df.loc[list(minutes.values()), "ws_raw"].max())
    cmap = "viridis"
    fig, axes = plt.subplots(1, len(minutes), figsize=(7 * len(minutes), 7))
    for ax, (label, minute) in zip(axes, minutes.items(), strict=True):
        row = plot_df.loc[minute]
        yaw = yaw_angles_for(df=plot_df.loc[[minute]], turbine_names=turbine_names, wtg=STEERING_WTG)
        fmodel.set(
            wind_data=TimeSeries(
                wind_directions=np.array([row["wd"]]),
                wind_speeds=np.array([k * row["ws_raw"]]),
                turbulence_intensities=np.array([row["ti"]]),
            ),
            yaw_angles=yaw,
        )
        # FLORIS reads plane bounds in its wind-aligned frame, so solve the whole farm and zoom afterwards
        plane = fmodel.calculate_horizontal_plane(
            height=HUB_HEIGHT_M, x_resolution=SNAPSHOT_RESOLUTION, y_resolution=SNAPSHOT_RESOLUTION
        )
        visualize_cut_plane(plane, ax=ax, min_speed=norm.vmin, max_speed=norm.vmax, cmap=cmap)
        plot_turbine_rotors(fmodel, ax=ax, yaw_angles=yaw)
        points = cone_points(
            apex_x=float(fmodel.layout_x[lidar]), apex_y=float(fmodel.layout_y[lidar]), yaw_pos_deg=row[LIDAR_YAW_COL]
        )
        probes = points.loc[points["segment_height_m"] == HUB_HEIGHT_M].groupby(["range_m", "side"])[["x", "y"]].mean()
        values = [
            measured.loc[minute, (range_m, HUB_HEIGHT_M, side)] / np.cos(np.deg2rad(HALF_ANGLE_DEG))
            for range_m, side in probes.index
        ]
        ax.scatter(probes["x"], probes["y"], c=values, cmap=cmap, norm=norm, s=60, edgecolors="white", zorder=3)
        ax.set_title(
            f"{label}: {minute:%H:%M} UTC, wd {row['wd']:.0f} deg\n"
            f"{STEERING_WTG} misalignment {row[MISALIGNMENT_COL]:+.1f} deg"
        )
        ax.set_xlim(*x_bounds)
        ax.set_ylim(*y_bounds)
        ax.set_aspect("equal")
        ax.set_xlabel("east [m]")
        ax.set_ylabel("north [m]")
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        label=f"FLORIS streamwise speed; probes: measured |LOS| / cos {HALF_ANGLE_DEG:g} deg [m/s]",
    )
    plot_path = out_dir / "flow_snapshots.png"
    logger.info("Writing: %s", plot_path)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    minute_df = load_minute_df()
    fmodel, turbine_names = load_hot_floris_model(unpack_local_meta_data())
    k = fit_toggle_off_k(minute_df=minute_df, fmodel=fmodel, turbine_names=turbine_names)

    # minutes without a T03 misalignment fall back to zero yaw so the whole window is plotted
    plot_df = minute_df.loc[minute_df[FREE_STREAM_COLS].notna().all(axis=1) & minute_df[LIDAR_YAW_COL].notna()].copy()
    plot_df[MISALIGNMENT_COL] = plot_df[MISALIGNMENT_COL].fillna(0)
    measured = load_zxtm_minute_df(start_dt=WINDOW_START, end_dt_excl=WINDOW_END_EXCL).reindex(plot_df.index)
    yawed = modelled_los_minutes(
        fmodel,
        turbine_names=turbine_names,
        df=plot_df,
        k=k,
        yaw_angles=yaw_angles_for(df=plot_df, turbine_names=turbine_names, wtg=STEERING_WTG),
    ).reindex(columns=measured.columns)
    unyawed = modelled_los_minutes(fmodel, turbine_names=turbine_names, df=plot_df, k=k).reindex(
        columns=measured.columns
    )
    matched = {
        MEASURED: measured,
        YAWED: yawed.where(measured.notna()),
        UNYAWED: unyawed.where(measured.notna()),
    }

    toggle_index = {}
    for toggle in ("off", "on"):
        filtered = filter_minutes(minute_df, toggle=toggle).index
        toggle_index[toggle] = filtered.intersection(plot_df.index)
        logger.info(
            "toggle-%s: %d minutes, %d dropped for missing %s",
            toggle,
            len(toggle_index[toggle]),
            len(filtered) - len(toggle_index[toggle]),
            LIDAR_YAW_COL,
        )

    results: dict = {"k": k, "segments": [], "hub_height_left_minus_right_shift_m_s": {}}
    for column in measured.columns:
        range_m, height_m, side = column
        results["segments"].append(
            {"range_m": range_m, "segment_height_m": height_m, "side": side}
            | {
                f"toggle_{toggle}": los_error_metrics(
                    measured=measured.loc[index, column], modelled=yawed.loc[index, column]
                )
                for toggle, index in toggle_index.items()
            }
        )
    for label, df in matched.items():
        diff = left_minus_right(df)
        shift = diff.loc[toggle_index["on"]].mean() - diff.loc[toggle_index["off"]].mean()
        results["hub_height_left_minus_right_shift_m_s"][label] = {f"{r:g}": float(v) for r, v in shift.items()}
    logger.info("hub-height left minus right shift: %s", results["hub_height_left_minus_right_shift_m_s"])
    results_path = out_dir / "results.json"
    logger.info("Writing: %s", results_path)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for toggle, index in toggle_index.items():
        plot_profiles(sources=matched, index=index, toggle=toggle, k=k, out_dir=out_dir)
    plot_hub_height_time_series(
        minute_df=minute_df, sources={MEASURED: measured, YAWED: yawed, UNYAWED: unyawed}, out_dir=out_dir
    )
    minutes = pick_snapshot_minutes(
        plot_df=plot_df, measured=measured, on_index=toggle_index["on"], off_index=toggle_index["off"]
    )
    plot_flow_snapshots(
        fmodel=fmodel,
        turbine_names=turbine_names,
        plot_df=plot_df,
        measured=measured,
        minutes=minutes,
        k=k,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    setup_logger(out_dir / f"{Path(__file__).stem}.log")
    main(out_dir)
