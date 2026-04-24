"""Goal is to make plots so that 10 minute, fastlog and LiDAR data can be inspected"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.lidar_helpers import load_zx_lidar_10min_data, load_zx_lidar_fl_data
from hot_open.scada_helpers import load_hot_10min_data
from hot_open.settings import get_out_dir
from scripts.logger import setup_logger

LOCAL_TEMPORARY_DIR = Path.home() / "temp" / "hill-of-towie-open-sourcing-2026" / "draft datapack"
logger = logging.getLogger(__name__)

# WTG 10-minute SCADA columns
WTG_10MIN_ACT_POWER_COL = "wtc_ActPower_mean"
WTG_10MIN_WIND_SPEED_COL = "wtc_AcWindSp_mean"
WTG_10MIN_NACEL_POS_COL = "wtc_NacelPos_mean"

# WTG fastlog columns
WTG_FL_ACT_POWER_COL = "ActPower_Value"
WTG_FL_WIND_SPEED_COL = "AcWindSp_AcWindSp"
WTG_FL_YAW_POS_COL = "YawPos_Value"

# ZX300 LiDAR 10-minute columns
ZX300_10MIN_WS_COL = "Horizontal Wind Speed (m/s) at 58m"
ZX300_10MIN_WD_COL = "Wind Direction (deg) at 58m"

# ZXTM LiDAR columns
ZXTM_10MIN_WS_COL = "FD Horizontal Wind Speed (m/s) at Hub Height at 208m"
ZXTM_10MIN_WD_COL = "Met Compass Bearing (deg)"
ZXTM_FL_WS_COL = "PD Horizontal Wind Speed (m/s) at Hub Height"

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
        wtg_fl_df[wtg][WTG_FL_YAW_POS_COL],
        drawstyle="steps-post",
        label=WTG_FL_YAW_POS_COL,
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


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    wtg_numbers = [3, 7]
    start_dt = pd.Timestamp("2026-02-19 22:30", tz="UTC")
    end_dt_excl = pd.Timestamp("2026-02-20 08:00", tz="UTC")

    # Load data for T3, T7 and both LiDARS from 2026-02-19 22:30 till 2026-02-19 08:00
    wtg_10min_df = load_hot_10min_data(
        data_dir=LOCAL_TEMPORARY_DIR,
        wtg_numbers=wtg_numbers,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
    )

    wtg_fl_df = load_hot_fl_data(
        data_dir=LOCAL_TEMPORARY_DIR / "turbine_fastlog" / "Filestore",
        wtg_numbers=wtg_numbers,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
    )

    for wtg_number in wtg_numbers:
        plot_wtg_10min_and_fastlog(
            wtg=f"T{wtg_number:02d}",
            wtg_10min_df=wtg_10min_df,
            wtg_fl_df=wtg_fl_df,
            start_dt=start_dt,
            end_dt_excl=end_dt_excl,
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

    # load reanalysis data
    era5_df = pd.read_parquet(
        Path(__name__).parent / "reanalysis_data" / "ERA5T_57.50N_-3.25E_100m_1hr_20260331.parquet"
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
        zx300_ws_col=ZX300_10MIN_WS_COL,
        zx300_wd_col=ZX300_10MIN_WD_COL,
        zxtm_ws_col=ZXTM_10MIN_WS_COL,
        zxtm_wd_col=ZXTM_10MIN_WD_COL,
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

    plot_lidar_10min_and_fastlog(
        zx300_10min_df=zx300_10min_df,
        zx300_fl_df=zx300_fl_df,
        zxtm_10min_df=zxtm_10min_df,
        zxtm_fl_df=zxtm_fl_df,
        zx300_ws_col=ZX300_10MIN_WS_COL,
        zx300_wd_col=ZX300_10MIN_WD_COL,
        zxtm_ws_col=ZXTM_10MIN_WS_COL,
        zxtm_wd_col=ZXTM_10MIN_WD_COL,
        zxtm_fl_ws_col=ZXTM_FL_WS_COL,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        out_dir=out_dir,
    )
