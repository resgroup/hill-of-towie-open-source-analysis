import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wake_steering_analysis.inspect_data import LOCAL_TEMPORARY_DIR

from hot_open.lidar_helpers import load_zx_lidar_10min_data
from hot_open.settings import get_cache_dir, get_out_dir
from scripts.logger import setup_logger

logger = logging.getLogger(__name__)


def plot_wake_dir_margin(
    *,
    upwind: str,
    ref: str,
    upwind_steer: pd.Series,
    ref_dir: pd.Series,
    first_wdir: float,
    last_wdir: float,
    out_dir: Path,
) -> None:
    wdir_min = first_wdir - 10
    wdir_max = last_wdir + 10

    df = pd.DataFrame({"ref_dir": ref_dir, "upwind_steer": upwind_steer}).dropna()
    df = df[(df["ref_dir"] >= wdir_min) & (df["ref_dir"] <= wdir_max)]

    if df.empty:
        logger.warning("No data for %s vs %s in range [%.0f, %.0f]", upwind, ref, wdir_min, wdir_max)
        return

    bins = np.arange(wdir_min, wdir_max + 1, 1)
    df["bin"] = pd.cut(df["ref_dir"], bins=bins)
    binned = df.groupby("bin", observed=True)["upwind_steer"].mean()
    bin_centers = [interval.mid for interval in binned.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["ref_dir"], df["upwind_steer"], alpha=0.3, s=10, label="scatter", color="tab:blue")
    ax.plot(bin_centers, binned.values, color="tab:orange", linewidth=2, marker="o", markersize=4, label="binned mean")
    ax.axvline(first_wdir, color="gray", linestyle="--", linewidth=1, label=f"wake sector [{first_wdir}°–{last_wdir}°]")
    ax.axvline(last_wdir, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(f"{ref} YawAngleMean [deg]")
    ax.set_ylabel(f"{upwind} mean_abs_wake_steer_command [deg]")
    ax.set_xlim(wdir_min, wdir_max)
    ax.legend()
    ax.grid(visible=True, alpha=0.3)
    fig.suptitle(f"Wake Steering vs Wind Direction\n{upwind} (upwind) vs {ref} (ref) yaw direction")

    plot_path = out_dir / f"wake_dir_margin_{upwind}_vs_{ref}_{first_wdir}_{last_wdir}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    logger.info("Saved plot to %s", plot_path)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    north_corrections = {
        "T01": -23.56751060485834,
        "T13": 1.2261276245117188,
    }

    scada_df = pd.read_parquet(get_cache_dir() / "unpack_scada_data" / "hot_dy_scada_df.parquet")
    scada_df = scada_df[scada_df.index > pd.Timestamp("2026-01-07 13:00:00", tz="UTC")]
    for upwind, ref, first_wdir, last_wdir in [
        ("T11", "ZX300_2428", 228, 255),
        ("T03", "T01", 192, 224),
        ("T02", "T01", 211, 241),
        ("T03", "T01", 260, 290),
    ]:
        upwind_steer = scada_df[scada_df["TurbineName"] == upwind]["mean_abs_wake_steer_command"]
        if ref == "ZX300_2428":
            lidar_unit_id = "2428"
            lidar_model = "ZX300"
            lidar_df = load_zx_lidar_10min_data(
                data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
                lidar_unit_id=lidar_unit_id,
                start_dt=pd.Timestamp("2026-01-01 00:00:00", tz="UTC"),
                end_dt_excl=pd.Timestamp("2026-05-01 00:00:00", tz="UTC"),
                remove_bad_values=True,
            )
            ref_dir = lidar_df["Wind Direction (deg) at 58m"]
        else:
            ref_dir = (scada_df[scada_df["TurbineName"] == ref]["YawAngleMean"] + north_corrections[ref]) % 360
        plot_wake_dir_margin(
            upwind=upwind,
            ref=ref,
            upwind_steer=upwind_steer,
            ref_dir=ref_dir,
            first_wdir=first_wdir,
            last_wdir=last_wdir,
            out_dir=out_dir,
        )
