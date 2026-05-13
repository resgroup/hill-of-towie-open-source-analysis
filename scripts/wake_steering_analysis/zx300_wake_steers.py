"""Goal is to label data where wake steering data (both toggle on and off)."""

import datetime as dt
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from hot_open.circular_math import circdiff_degrees
from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_fl_data
from hot_open.settings import get_filestore_dir, get_out_dir
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.inspect_data import (
    LOCAL_TEMPORARY_DIR,
    NORTH_CORRECTIONS,
    WTG_FL_ACT_POWER_COL,
    WTG_FL_YAW_POS_COL,
)

logger = logging.getLogger(__name__)
SMOOTHING_WINDOW = 600


def _shade_toggle(ax, *, steer_df, toggle_col):
    ylow, yhigh = ax.get_ylim()
    ax.fill_between(
        steer_df.index, ylow, yhigh, where=steer_df[toggle_col] == 0, color="red", alpha=0.05, label="steering off"
    )
    ax.fill_between(
        steer_df.index, ylow, yhigh, where=steer_df[toggle_col] == 1, color="green", alpha=0.05, label="steering on"
    )
    ax.set_ylim(ylow, yhigh)


def _finalize_and_save(axes, *, plot_start, plot_end, title, plot_dir):
    for ax in axes:
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=True)
    plt.xlim(plot_start, plot_end)
    plt.xticks(rotation=90)
    plt.xlabel("timestamp")
    plt.suptitle(title)
    plt.tight_layout()
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f"{title.replace(':', '')}.png", bbox_inches="tight")
    plt.close()
    logger.info("finished %s %s", plot_start, title)


def _add_smoothed(df, *, col, new_col, min_periods=SMOOTHING_WINDOW // 2):
    df[new_col] = df[col].rolling(window=SMOOTHING_WINDOW, min_periods=min_periods).mean()


def _slice_period(df, *, start, end):
    return df[(df.index >= start) & (df.index < end)]


def _plot_wake_steering_period(
    *,
    plot_ref_df,
    plot_steer_df,
    plot_dep_df,
    plot_zx300_df,
    ref_name,
    steering_name,
    dependent_turbine_name,
    wd_col,
    yawpos_col,
    toggle_col,
    smoothed_pw_col,
    plot_start,
    plot_end,
    out_dir,
):
    plot_dir = out_dir / f"{steering_name}-{dependent_turbine_name}-{ref_name}"

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axid = 0
    ax = axes[axid]

    ax.plot(
        plot_ref_df.index, plot_ref_df[wd_col], label=f"{ref_name} wind direction", color="C0", linestyle=":", alpha=0.5
    )
    ax.plot(plot_ref_df.index, plot_ref_df[yawpos_col], label=f"{ref_name} yaw position", color="C0")
    ax.plot(
        plot_steer_df.index,
        plot_steer_df[wd_col],
        label=f"{steering_name} wind direction",
        color="C1",
        linestyle=":",
        alpha=0.5,
    )
    ax.plot(plot_steer_df.index, plot_steer_df[yawpos_col], label=f"{steering_name} yaw position", color="C1")
    ax.plot(
        plot_dep_df.index,
        plot_dep_df[wd_col],
        label=f"{dependent_turbine_name} wind direction",
        color="C2",
        linestyle=":",
        alpha=0.5,
    )
    ax.plot(plot_dep_df.index, plot_dep_df[yawpos_col], label=f"{dependent_turbine_name} yaw position", color="C2")
    _shade_toggle(ax, steer_df=plot_steer_df, toggle_col=toggle_col)
    ax.set_ylabel("direction [degN]")

    axid += 1
    ax = axes[axid]
    ax.plot(plot_ref_df.index, plot_ref_df[smoothed_pw_col], label=f"{ref_name} smoothed power")
    ax.plot(plot_steer_df.index, plot_steer_df[smoothed_pw_col], label=f"{steering_name} smoothed power")
    ax.plot(plot_dep_df.index, plot_dep_df[smoothed_pw_col], label=f"{dependent_turbine_name} smoothed power")
    _shade_toggle(ax, steer_df=plot_steer_df, toggle_col=toggle_col)
    ax.set_ylabel("power [kW]")

    axid += 1
    ax = axes[axid]
    zx300_hh_ws_col='Horizontal Wind Speed (m/s) at 58m'
    zx300_hh_wd_col = 'Wind Direction (deg) at 58m'
    ax_twin = ax.twinx()
    ax.plot(plot_zx300_df.index, plot_zx300_df[zx300_hh_ws_col], label=zx300_hh_ws_col)
    ax_twin.plot(plot_zx300_df.index, plot_zx300_df[zx300_hh_wd_col], label=zx300_hh_wd_col, color="C1")
    _shade_toggle(ax, steer_df=plot_steer_df, toggle_col=toggle_col)
    ax.set_ylabel("LiDAR wind speed [m/s]")
    ax_twin.set_ylabel("LiDAR wind direction [degN]")
    for _ax in (ax, ax_twin):
        _ax.yaxis.set_major_locator(ticker.LinearLocator(6))

    title = f"{steering_name} steering for {dependent_turbine_name} {plot_start.strftime('%Y-%m-%d %H:%M')} to {plot_end.strftime('%H:%M')}"
    _finalize_and_save([*axes, ax_twin], plot_start=plot_start, plot_end=plot_end, title=title, plot_dir=plot_dir)

    # plot diffs
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    axid = 0
    ax = axes[axid]

    steering_yawerr = (
        circdiff_degrees(plot_steer_df[wd_col], plot_steer_df[yawpos_col])
        .rolling(window=SMOOTHING_WINDOW, min_periods=SMOOTHING_WINDOW // 2)
        .mean()
    )
    ax.plot(
        plot_ref_df.index,
        circdiff_degrees(plot_ref_df[wd_col], plot_ref_df[yawpos_col])
        .rolling(window=SMOOTHING_WINDOW, min_periods=SMOOTHING_WINDOW // 2)
        .mean(),
        label=f"{ref_name} yaw error",
    )
    ax.plot(plot_steer_df.index, steering_yawerr, label=f"{steering_name} yaw error")
    ax.plot(
        plot_dep_df.index,
        circdiff_degrees(plot_dep_df[wd_col], plot_dep_df[yawpos_col])
        .rolling(window=SMOOTHING_WINDOW, min_periods=SMOOTHING_WINDOW // 2)
        .mean(),
        label=f"{dependent_turbine_name} yaw error",
    )
    _shade_toggle(ax, steer_df=plot_steer_df, toggle_col=toggle_col)
    ax.set_ylabel("direction [degN]")

    axid += 1
    ax = axes[axid]
    ax.plot(
        plot_steer_df.index,
        plot_steer_df[smoothed_pw_col]
        - plot_ref_df[smoothed_pw_col]
        - (plot_steer_df[smoothed_pw_col] - plot_ref_df[smoothed_pw_col]).mean(),
        label=f"{steering_name} - {ref_name} smoothed power",
        color="C1",
    )
    ax.plot(
        plot_dep_df.index,
        plot_dep_df[smoothed_pw_col]
        - plot_ref_df[smoothed_pw_col]
        - (plot_dep_df[smoothed_pw_col] - plot_ref_df[smoothed_pw_col]).mean(),
        label=f"{dependent_turbine_name} - {ref_name} smoothed power",
        color="C2",
    )
    plot_ser = (
        plot_steer_df[smoothed_pw_col] + plot_dep_df[smoothed_pw_col] - 2 * plot_ref_df[smoothed_pw_col]
    )  # TODO try normalizing
    ax.plot(
        plot_dep_df.index,
        plot_ser - plot_ser.mean(),
        label=f"{steering_name}+{dependent_turbine_name} - 2*{ref_name} smoothed power",
        color="black",
    )
    ax.plot(
        plot_steer_df.index,
        steering_yawerr.abs() * 20,
        label=f"{steering_name} abs yaw error * 20",
        color="grey",
        alpha=0.5,
    )
    _shade_toggle(ax, steer_df=plot_steer_df, toggle_col=toggle_col)
    ax.set_ylabel("power diff [kW]")

    if False:
        axid += 1
        ax = axes[axid]
        ax.plot(
            plot_zxtm_df.index,
            plot_zxtm_df[smoothed_right_ws_col] - plot_zxtm_df[smoothed_left_ws_col],  # TODO try normalizing
            label="ZXTM smoothed right - left ws",
        )
        ax.plot(plot_steer_df.index, steering_yawerr / 5, label=f"{steering_name} yaw error / 5", color="grey", alpha=0.5)
        _shade_toggle(ax, steer_df=plot_steer_df, toggle_col=toggle_col)
        ax.set_ylabel("wind speed diff [m/s]")

    title = f"{steering_name} steering for {dependent_turbine_name} {plot_start.strftime('%Y-%m-%d %H:%M')} to {plot_end.strftime('%H:%M')} diffs"
    _finalize_and_save(axes, plot_start=plot_start, plot_end=plot_end, title=title, plot_dir=plot_dir)


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    campaign_config_csv_dir = Path(__file__).parent / "controller_config"
    campaign_default_steering_table = pd.read_csv(campaign_config_csv_dir / "wake-steering-lookup.csv").rename(
        columns={
            "device_name": "TurbineName",
            "wind_direction_degrees": "WindDirection_deg",
            "yaw_offset_degrees": "YawOffset_deg",
            "dependent_device_name": "DependentTurbineNames",
        }
    )

    wtg_numbers = [11,13, 14]
    # Jan 14, Feb 23, Mar 20, Apr 4
    start_dt = pd.Timestamp("2026-01-14 00:00", tz="UTC")  # pd.Timestamp("2026-01-07 13:00", tz="UTC")
    end_dt_excl = pd.Timestamp("2026-01-15 00:00", tz="UTC")  # pd.Timestamp("2026-05-01 08:00", tz="UTC")

    toggle_col = "computed_driver_post_processed_toggle_state"
    yawpos_col = "computed_driver_pre_processed_yaw_direction_true_degrees"
    wd_col = "computed_core_post_processed_consensus_wind_direction_true_degrees"
    wake_steer_col = "computed_core_post_processed_core_wake_steering_offset_degrees"

    wtg_fl_df = load_hot_fl_data(
        data_dir=get_filestore_dir(),
        wtg_numbers=wtg_numbers,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        extra_tags=[toggle_col, yawpos_col, wd_col, wake_steer_col],
    )
    for wtg, correction in NORTH_CORRECTIONS.items():
        if wtg in wtg_fl_df.columns.get_level_values(0):
            wtg_fl_df[(wtg, WTG_FL_YAW_POS_COL)] = (wtg_fl_df[(wtg, WTG_FL_YAW_POS_COL)] + correction) % 360

    ref_name = "T13"
    steering_name = "T11"
    dependent_turbine_name = "T14"
    steer_config = campaign_default_steering_table[
        (campaign_default_steering_table["TurbineName"] == steering_name)
        & (campaign_default_steering_table["DependentTurbineNames"] == dependent_turbine_name)
    ]
    ref_df = wtg_fl_df[ref_name].copy()
    steering_df = wtg_fl_df[steering_name].copy()
    dependent_df = wtg_fl_df[dependent_turbine_name].copy()

    zx300_fl_df = load_zx_lidar_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="2428",
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )

    pw_col = WTG_FL_ACT_POWER_COL
    smoothed_pw_col = "smoothed_pw"
    for df in (ref_df, steering_df, dependent_df):
        _add_smoothed(df, col=pw_col, new_col=smoothed_pw_col)

    toggle_switch = steering_df[toggle_col].diff().fillna(0) != 0
    first_toggle_off = steering_df.index[toggle_switch & (steering_df[toggle_col] == 0)][0]
    minutes_to_include_before = 10
    minutes_to_include_after = 10
    toggle_period_minutes = 50 * 2
    toggle_periods_in_one_plot = 2
    plot_start = first_toggle_off - dt.timedelta(minutes=minutes_to_include_before)
    plot_duration = (
        dt.timedelta(minutes=minutes_to_include_before)
        + dt.timedelta(minutes=minutes_to_include_after)
        + dt.timedelta(minutes=toggle_period_minutes * toggle_periods_in_one_plot)
    )
    plot_timedelta = dt.timedelta(minutes=toggle_period_minutes * toggle_periods_in_one_plot)
    while plot_start < (steering_df.index.max() - plot_duration):
        plot_end = plot_start + plot_duration

        plot_ref_df = _slice_period(ref_df, start=plot_start, end=plot_end)
        plot_steer_df = _slice_period(steering_df, start=plot_start, end=plot_end)
        plot_dep_df = _slice_period(dependent_df, start=plot_start, end=plot_end)
        plot_zx300_df = _slice_period(zx300_fl_df, start=plot_start, end=plot_end)

        # if there is no interesting steering, skip
        if ((plot_steer_df[wake_steer_col].abs() > 10) & (plot_steer_df[toggle_col] == 1)).sum() < (1800 / 3):
            plot_start += plot_timedelta
            continue

        _plot_wake_steering_period(
            plot_ref_df=plot_ref_df,
            plot_steer_df=plot_steer_df,
            plot_dep_df=plot_dep_df,
            plot_zx300_df=plot_zx300_df,
            ref_name=ref_name,
            steering_name=steering_name,
            dependent_turbine_name=dependent_turbine_name,
            wd_col=wd_col,
            yawpos_col=yawpos_col,
            toggle_col=toggle_col,
            smoothed_pw_col=smoothed_pw_col,
            plot_start=plot_start,
            plot_end=plot_end,
            out_dir=out_dir,
        )
        plot_start += plot_timedelta
        msg = f"finished {plot_start=}"
        logger.info(msg)
