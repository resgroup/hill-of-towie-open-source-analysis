"""Goal is to make plots so that 10 minute, fastlog and LiDAR data can be inspected"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_10min_data, load_zx_lidar_fl_data
from hot_open.scada_helpers import load_hot_10min_data
from hot_open.settings import get_data_dir, get_out_dir
from scripts.logger import setup_logger

LOCAL_TEMPORARY_DIR = get_data_dir()
logger = logging.getLogger(__name__)

# WTG 10-minute SCADA columns
WTG_10MIN_ACT_POWER_COL = "wtc_ActPower_mean"
WTG_10MIN_WIND_SPEED_COL = "wtc_AcWindSp_mean"
WTG_10MIN_NACEL_POS_COL = "wtc_NacelPos_mean"
WTG_10MIN_YAW_OPE_COUNTS_COL = "wtc_ScYawOpe_counts"

# WTG fastlog columns
WTG_FL_ACT_POWER_COL = "ActPower_Value"
WTG_FL_WIND_SPEED_COL = "AcWindSp_AcWindSp"
WTG_FL_YAW_POS_COL = "YawPos_Value"
WTG_FL_NORTHED_YAW_POS_COL = "northed YawPos_Value"

# ZX300 LiDAR 10-minute columns
ZX300_WS_COL = "Horizontal Wind Speed (m/s) at 58m"
ZX300_WD_COL = "Wind Direction (deg) at 58m"

# ZXTM LiDAR columns
ZXTM_10MIN_WS_COL = "FD Horizontal Wind Speed (m/s) at Hub Height at 208m"
ZXTM_FL_WS_COL = "PD Horizontal Wind Speed (m/s) at Hub Height"
ZXTM_WD_COL = "Met Compass Bearing (deg)"

# ERA5 reanalysis columns
ERA5_WS_COL = "100_m_hws_mean_mps"
ERA5_WD_COL = "100_m_hwd_mean_deg-n_true"


def plot_wtg_10min_and_fastlog(
    *,
    wtg: str,
    wtg_10min_df: pd.DataFrame,
    wtg_fl_df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(
        wtg_10min_df.index,
        wtg_10min_df[wtg][WTG_10MIN_ACT_POWER_COL],
        drawstyle="steps-post",
        label=WTG_10MIN_ACT_POWER_COL,
        linewidth=1.5,
    )
    axes[0].plot(
        wtg_fl_df.index,
        wtg_fl_df[wtg][WTG_FL_ACT_POWER_COL],
        drawstyle="steps-post",
        label=WTG_FL_ACT_POWER_COL,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    axes[1].plot(
        wtg_10min_df.index,
        wtg_10min_df[wtg][WTG_10MIN_WIND_SPEED_COL],
        drawstyle="steps-post",
        label=WTG_10MIN_WIND_SPEED_COL,
        linewidth=1.5,
    )
    axes[1].plot(
        wtg_fl_df.index,
        wtg_fl_df[wtg][WTG_FL_WIND_SPEED_COL],
        drawstyle="steps-post",
        label=WTG_FL_WIND_SPEED_COL,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    axes[2].plot(
        wtg_10min_df.index,
        wtg_10min_df[wtg][WTG_10MIN_NACEL_POS_COL],
        drawstyle="steps-post",
        label=WTG_10MIN_NACEL_POS_COL,
        linewidth=1.5,
    )
    axes[2].plot(
        wtg_fl_df.index,
        wtg_fl_df[wtg][WTG_FL_NORTHED_YAW_POS_COL],
        drawstyle="steps-post",
        label=WTG_FL_NORTHED_YAW_POS_COL,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )

    axes[0].set_ylabel("Active Power [kW]")
    axes[1].set_ylabel("Wind Speed [m/s]")
    axes[2].set_ylabel("Nacelle Position [deg]")
    axes[2].set_xlabel("timestamp")
    axes[2].tick_params(axis="x", rotation=90)
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
        ax.grid(visible=True, alpha=0.3)
    date_range_str = f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt_excl.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(f"{wtg}\n{date_range_str}")

    plot_path = out_dir / f"10min_vs_fastlog_{wtg}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path)
    plt.close(fig)


def plot_yaw_ope_counts_check(
    *,
    wtg: str,
    wtg_10min_df: pd.DataFrame,
    wtg_fl_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    subdir = out_dir / "yaw_ope_counts"
    subdir.mkdir(exist_ok=True)

    hours = pd.DatetimeIndex(wtg_10min_df.index).floor("h").unique()
    for hour_start in hours:
        hour_end = hour_start + pd.Timedelta(hours=1)
        df_10min = wtg_10min_df[(wtg_10min_df.index >= hour_start) & (wtg_10min_df.index < hour_end)]
        df_fl = wtg_fl_df[(wtg_fl_df.index >= hour_start) & (wtg_fl_df.index < hour_end)]
        if df_10min.empty or df_fl.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

        axes[0].plot(
            df_fl.index,
            df_fl[wtg][WTG_FL_NORTHED_YAW_POS_COL],
            drawstyle="steps-post",
            label=WTG_FL_NORTHED_YAW_POS_COL,
            linewidth=0.8,
        )
        for t in df_10min.index:
            axes[0].axvline(t, color="gray", alpha=0.25, linewidth=0.8, linestyle="--")
        axes[0].set_ylabel("Northed Yaw Pos [deg]")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
        axes[0].grid(visible=True, alpha=0.3)

        axes[1].plot(
            df_10min.index,
            df_10min[wtg][WTG_10MIN_YAW_OPE_COUNTS_COL],
            drawstyle="steps-post",
            label=WTG_10MIN_YAW_OPE_COUNTS_COL,
            linewidth=1.5,
        )
        axes[1].set_ylabel("Yaw Op Count")
        axes[1].set_xlabel("timestamp")
        axes[1].tick_params(axis="x", rotation=90)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
        axes[1].grid(visible=True, alpha=0.3)

        hour_str = hour_start.strftime("%Y-%m-%d_%H")
        fig.suptitle(f"{wtg} Yaw Ope Counts Check\n{hour_str} UTC")

        plot_path = subdir / f"yaw_ope_counts_{wtg}_{hour_str}UTC.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", plot_path)
        plt.close(fig)


def _plot_dynamic_yaw_control_tags(
    *,
    wtg: str,
    wtg_fl_df: pd.DataFrame,
    toggle_col: str,
    wake_steer_col: str,
    wd_col: str,
    yawpos_col: str,
    yawtarget_col: str,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fl = wtg_fl_df[wtg]

    ax0 = axes[0]
    ax0_twin = ax0.twinx()
    ax0.plot(fl.index, fl[toggle_col], drawstyle="steps-post", label=toggle_col, linewidth=1.5, color="tab:blue")
    ax0_twin.plot(
        fl.index, fl[wake_steer_col], drawstyle="steps-post", label=wake_steer_col, linewidth=1.5, color="tab:orange"
    )
    ax0.set_ylabel("Toggle State")
    ax0_twin.set_ylabel("Wake Steering Offset [deg]")
    ax0.legend(loc="upper left", bbox_to_anchor=(1.08, 1), fontsize="small")
    ax0_twin.legend(loc="upper left", bbox_to_anchor=(1.08, 0.85), fontsize="small")

    axes[1].plot(fl.index, fl[wd_col], drawstyle="steps-post", label=wd_col, linewidth=1.5)
    axes[1].plot(
        fl.index, fl[yawpos_col], drawstyle="steps-post", label=yawpos_col, linestyle="--", linewidth=0.8, alpha=0.7
    )
    axes[1].set_ylabel("Direction [deg]")

    axes[2].plot(fl.index, fl[yawtarget_col], drawstyle="steps-post", label=yawtarget_col, linewidth=1.5)
    axes[2].plot(
        fl.index,
        fl[WTG_FL_YAW_POS_COL],
        drawstyle="steps-post",
        label=WTG_FL_YAW_POS_COL,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
    )
    axes[2].set_ylabel("Yaw [deg]")
    axes[2].set_xlabel("timestamp")
    axes[2].tick_params(axis="x", rotation=90)

    for ax in axes[1:]:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    for ax in axes:
        ax.grid(visible=True, alpha=0.3)

    date_range_str = f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt_excl.strftime('%Y-%m-%d %H:%M')} UTC"
    fig.suptitle(f"{wtg} Dynamic Yaw Control Tags\n{date_range_str}")

    plot_path = out_dir / f"dynamic_yaw_control_tags_{wtg}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path)
    plt.close(fig)


def plot_lidar_10min_and_fastlog(
    *,
    zx300_10min_df: pd.DataFrame,
    zx300_fl_df: pd.DataFrame,
    zxtm_10min_df: pd.DataFrame,
    zxtm_fl_df: pd.DataFrame,
    zx300_ws_col: str,
    zx300_wd_col: str,
    zxtm_ws_col: str,
    zxtm_wd_col: str,
    zxtm_fl_ws_col: str,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    out_dir: Path,
) -> None:
    date_range_str = f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt_excl.strftime('%Y-%m-%d %H:%M')} UTC"

    for lidar_name, ten_min_df, fl_df, ws_col, wd_col, fl_ws_col in [
        ("ZX300", zx300_10min_df, zx300_fl_df, zx300_ws_col, zx300_wd_col, zx300_ws_col),
        ("ZXTM", zxtm_10min_df, zxtm_fl_df, zxtm_ws_col, zxtm_wd_col, zxtm_fl_ws_col),
    ]:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(
            ten_min_df.index,
            ten_min_df[ws_col],
            drawstyle="steps-post",
            label=f"10min {ws_col}",
            linewidth=1.5,
        )
        axes[0].plot(
            fl_df.index,
            fl_df[fl_ws_col],
            drawstyle="steps-post",
            label=f"fastlog {fl_ws_col}",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        axes[1].plot(
            ten_min_df.index,
            ten_min_df[wd_col],
            drawstyle="steps-post",
            label=f"10min {wd_col}",
            linewidth=1.5,
        )
        axes[1].plot(
            fl_df.index,
            fl_df[wd_col],
            drawstyle="steps-post",
            label=f"fastlog {wd_col}",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )

        axes[0].set_ylabel("Wind Speed [m/s]")
        axes[1].set_ylabel("Wind Direction [deg]")
        axes[1].set_xlabel("timestamp")
        axes[1].tick_params(axis="x", rotation=90)
        for ax in axes:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
            ax.grid(visible=True, alpha=0.3)
        fig.suptitle(f"{lidar_name}\n{date_range_str}")

        plot_path = out_dir / f"10min_vs_fastlog_{lidar_name}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        logger.info("Saved plot to %s", plot_path)
        plt.close(fig)


def plot_wind_speed_and_direction_comparison(
    *,
    wtg_10min_df: pd.DataFrame,
    wtg_numbers: list[int],
    zx300_10min_df: pd.DataFrame,
    zxtm_10min_df: pd.DataFrame,
    zx300_ws_col: str,
    zx300_wd_col: str,
    zxtm_ws_col: str,
    zxtm_wd_col: str,
    era5_df: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for wtg_number in wtg_numbers:
        wtg = f"T{wtg_number:02d}"
        axes[0].plot(
            wtg_10min_df.index,
            wtg_10min_df[wtg][WTG_10MIN_WIND_SPEED_COL],
            drawstyle="steps-post",
            label=f"{wtg} {WTG_10MIN_WIND_SPEED_COL}",
            linewidth=1.5,
        )
        axes[1].plot(
            wtg_10min_df.index,
            wtg_10min_df[wtg][WTG_10MIN_NACEL_POS_COL],
            drawstyle="steps-post",
            label=f"{wtg} {WTG_10MIN_NACEL_POS_COL}",
            linewidth=1.5,
        )

    axes[0].plot(
        zx300_10min_df.index,
        zx300_10min_df[zx300_ws_col],
        drawstyle="steps-post",
        label=f"ZX300 {zx300_ws_col}",
        alpha=0.7,
    )
    axes[0].plot(
        zxtm_10min_df.index,
        zxtm_10min_df[zxtm_ws_col],
        drawstyle="steps-post",
        label=f"ZXTM {zxtm_ws_col}",
        alpha=0.7,
    )
    axes[0].plot(
        era5_df.index,
        era5_df[ERA5_WS_COL],
        drawstyle="steps-post",
        label=f"ERA5 {ERA5_WS_COL}",
        linestyle=":",
        alpha=0.7,
    )
    axes[1].plot(
        zx300_10min_df.index,
        zx300_10min_df[zx300_wd_col],
        drawstyle="steps-post",
        label=f"ZX300 {zx300_wd_col}",
        alpha=0.7,
    )
    axes[1].plot(
        zxtm_10min_df.index,
        zxtm_10min_df[zxtm_wd_col],
        drawstyle="steps-post",
        label=f"ZXTM {zxtm_wd_col}",
        alpha=0.7,
    )
    axes[1].plot(
        era5_df.index,
        era5_df[ERA5_WD_COL],
        drawstyle="steps-post",
        label=f"ERA5 {ERA5_WD_COL}",
        linestyle=":",
        alpha=0.7,
    )

    axes[0].set_ylabel("Wind Speed [m/s]")
    axes[1].set_ylabel("Wind Direction [deg]")
    axes[1].set_xlabel("timestamp")
    axes[1].tick_params(axis="x", rotation=90)
    for ax in axes:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
        ax.grid(visible=True, alpha=0.3)
    date_range_str = f"{start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt_excl.strftime('%Y-%m-%d %H:%M')} UTC"
    wtgs_str = ", ".join(f"T{n:02d}" for n in wtg_numbers)
    fig.suptitle(f"Wind Speed and Direction Comparison: {wtgs_str} vs LiDARs\n{date_range_str}")

    plot_path = out_dir / "wind_speed_and_direction_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path)
    plt.close(fig)


NORTH_CORRECTIONS = {
    "T03": -177.46022644042966,
    "T07": 39.44209289550781,
    "ZXTM": -131,
}

if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    wtg_numbers = [3, 7]
    start_dt = pd.Timestamp("2026-02-19 22:30", tz="UTC")
    end_dt_excl = pd.Timestamp("2026-02-20 07:00", tz="UTC")

    # Load data for T3, T7 and both LiDARS from 2026-02-19 22:30 till 2026-02-19 08:00
    wtg_10min_df = load_hot_10min_data(
        data_dir=LOCAL_TEMPORARY_DIR,
        wtg_numbers=wtg_numbers,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
    )
    for wtg, correction in NORTH_CORRECTIONS.items():
        if wtg in wtg_10min_df.columns.get_level_values(0):
            wtg_10min_df[(wtg, WTG_10MIN_NACEL_POS_COL)] = (
                wtg_10min_df[(wtg, WTG_10MIN_NACEL_POS_COL)] + correction
            ) % 360

    dy_toggle_col = "computed_driver_post_processed_toggle_state"
    dy_wake_steer_col = "computed_core_post_processed_core_wake_steering_offset_degrees"
    dy_wd_col = "computed_core_post_processed_consensus_wind_direction_true_degrees"
    dy_northed_yawpos_col = "computed_driver_pre_processed_yaw_direction_true_degrees"
    dy_yawtarget_col = "computed_driver_post_processed_yaw_target_degrees"

    wtg_fl_df = load_hot_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "turbine_fastlog" / "Filestore",
        wtg_numbers=wtg_numbers,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        extra_tags=[dy_toggle_col, dy_northed_yawpos_col, dy_yawtarget_col, dy_wd_col, dy_wake_steer_col],
    )

    for wtg, correction in NORTH_CORRECTIONS.items():
        if wtg in wtg_fl_df.columns.get_level_values(0):
            wtg_fl_df[(wtg, WTG_FL_NORTHED_YAW_POS_COL)] = (wtg_fl_df[(wtg, WTG_FL_YAW_POS_COL)] + correction) % 360

    for wtg_number in wtg_numbers:
        wtg = f"T{wtg_number:02d}"
        plot_wtg_10min_and_fastlog(
            wtg=wtg,
            wtg_10min_df=wtg_10min_df,
            wtg_fl_df=wtg_fl_df,
            start_dt=start_dt,
            end_dt_excl=end_dt_excl,
            out_dir=out_dir,
        )
        _plot_dynamic_yaw_control_tags(
            wtg=wtg,
            wtg_fl_df=wtg_fl_df,
            toggle_col=dy_toggle_col,
            wake_steer_col=dy_wake_steer_col,
            wd_col=dy_wd_col,
            yawpos_col=dy_northed_yawpos_col,
            yawtarget_col=dy_yawtarget_col,
            start_dt=start_dt,
            end_dt_excl=end_dt_excl,
            out_dir=out_dir,
        )
        plot_yaw_ope_counts_check(
            wtg=wtg,
            wtg_10min_df=wtg_10min_df,
            wtg_fl_df=wtg_fl_df,
            out_dir=out_dir,
        )

    zx300_10min_df = load_zx_lidar_10min_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="2428",
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )

    zxtm_10min_df = load_zx_lidar_10min_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="5060",
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )
    zxtm_10min_df[ZXTM_WD_COL] = (zxtm_10min_df[ZXTM_WD_COL] + NORTH_CORRECTIONS["ZXTM"]) % 360

    # load reanalysis data
    era5_df = pd.read_parquet(
        Path(__file__).parent / "reanalysis_data" / "ERA5T_57.50N_-3.25E_100m_1hr_20260331.parquet"
    )
    era5_df = era5_df[
        (era5_df.index >= (start_dt - pd.Timedelta(minutes=50)))
        & (era5_df.index < (end_dt_excl + pd.Timedelta(minutes=50)))
    ]

    plot_wind_speed_and_direction_comparison(
        wtg_10min_df=wtg_10min_df,
        wtg_numbers=wtg_numbers,
        zx300_10min_df=zx300_10min_df,
        zxtm_10min_df=zxtm_10min_df,
        zx300_ws_col=ZX300_WS_COL,
        zx300_wd_col=ZX300_WD_COL,
        zxtm_ws_col=ZXTM_10MIN_WS_COL,
        zxtm_wd_col=ZXTM_WD_COL,
        era5_df=era5_df,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        out_dir=out_dir,
    )

    zx300_fl_df = load_zx_lidar_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="2428",
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )
    zxtm_fl_df = load_zx_lidar_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
        lidar_unit_id="5060",
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        remove_bad_values=True,
    )
    zxtm_fl_df[ZXTM_WD_COL] = (zxtm_fl_df[ZXTM_WD_COL] + NORTH_CORRECTIONS["ZXTM"]) % 360

    plot_lidar_10min_and_fastlog(
        zx300_10min_df=zx300_10min_df,
        zx300_fl_df=zx300_fl_df,
        zxtm_10min_df=zxtm_10min_df,
        zxtm_fl_df=zxtm_fl_df,
        zx300_ws_col=ZX300_WS_COL,
        zx300_wd_col=ZX300_WD_COL,
        zxtm_ws_col=ZXTM_10MIN_WS_COL,
        zxtm_wd_col=ZXTM_WD_COL,
        zxtm_fl_ws_col=ZXTM_FL_WS_COL,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        out_dir=out_dir,
    )
