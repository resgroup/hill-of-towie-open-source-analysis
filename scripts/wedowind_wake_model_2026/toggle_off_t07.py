"""FLORIS toggle-off baseline: predict T07 power from T01 free-stream inputs, calibrated to unwaked T03.

See ``hot_floris.py`` for the model and free-stream inputs. T03 has its measured yaw misalignment, every other
turbine zero yaw.

Run with ``uv sync --group wedowind`` first.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from hot_open.settings import get_out_dir
from hot_open.unpack import unpack_local_meta_data
from scripts.logger import setup_logger
from scripts.wedowind_wake_model_2026.hot_floris import (
    CALIBRATION_WTG,
    FREE_STREAM_COLS,
    FREE_STREAM_WTG,
    PREDICTION_WTG,
    STEERING_SECTOR_DEG,
    STEERING_WTG,
    WINDOW_END_EXCL,
    WINDOW_START,
    error_metrics,
    filter_minutes,
    fit_toggle_off_k,
    load_hot_floris_model,
    load_minute_df,
    plot_scatter,
    run_floris_minutes,
    shade_toggle_on,
    yaw_angles_for,
)

logger = logging.getLogger(__name__)


def plot_time_series(
    *, minute_df: pd.DataFrame, modelled_df: pd.DataFrame, filtered_df: pd.DataFrame, k: float, out_dir: Path
) -> None:
    """Plot the FLORIS inputs, then measured vs modelled T03 and T07 power, with the toggle-off bias in the legend."""
    modelled_df = modelled_df.reindex(minute_df.index)

    fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
    ax_ws, ax_ti, ax_wd, *power_axes = axes
    for ax in axes:
        shade_toggle_on(ax, minute_df=minute_df)

    ax_ws.plot(minute_df.index, k * minute_df["ws_raw"], drawstyle="steps-post", label=f"{k:.3f} x {FREE_STREAM_WTG}")
    ax_ws.set_ylabel("wind speed [m/s]")
    ax_ti.plot(minute_df.index, minute_df["ti"], drawstyle="steps-post", label=f"{FREE_STREAM_WTG} anemometer")
    ax_ti.set_ylabel("TI [-]")
    ax_wd.plot(minute_df.index, minute_df["wd"], drawstyle="steps-post", label=f"{FREE_STREAM_WTG} DY consensus")
    for i, sector_edge in enumerate(STEERING_SECTOR_DEG):
        ax_wd.axhline(
            sector_edge, color="black", linestyle=":", linewidth=0.8, label="filter sector" if i == 0 else None
        )
    ax_wd.set_ylabel("wind direction [deg]")

    for ax, wtg in zip(power_axes, [CALIBRATION_WTG, PREDICTION_WTG], strict=True):
        bias = round((modelled_df.loc[filtered_df.index, wtg] - filtered_df[f"{wtg}_power"]).mean())
        ax.plot(minute_df.index, minute_df[f"{wtg}_power"], drawstyle="steps-post", label="measured")
        ax.plot(
            modelled_df.index,
            modelled_df[wtg],
            drawstyle="steps-post",
            linestyle="--",
            label=f"FLORIS (toggle-off bias {bias:+d} kW)",
        )
        ax.set_ylabel(f"{wtg} power [kW]")

    for ax in axes:
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    axes[-1].set_xlabel("timestamp (UTC)")
    fig.suptitle(
        f"FLORIS inputs and power, {STEERING_WTG} measured yaw, k fitted on {len(filtered_df)} "
        f"toggle-off {CALIBRATION_WTG} minutes\n{WINDOW_START:%Y-%m-%d %H:%M} to {WINDOW_END_EXCL:%H:%M} UTC"
    )
    plot_path = out_dir / "time_series_slide_window.png"
    logger.info("Writing: %s", plot_path)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    minute_df = load_minute_df()
    filtered_df = filter_minutes(minute_df, toggle="off")
    fmodel, turbine_names = load_hot_floris_model(unpack_local_meta_data())
    k = fit_toggle_off_k(minute_df=minute_df, fmodel=fmodel, turbine_names=turbine_names)

    def run(df: pd.DataFrame) -> pd.DataFrame:
        yaw = yaw_angles_for(df=df, turbine_names=turbine_names, wtg=STEERING_WTG)
        return run_floris_minutes(fmodel, turbine_names=turbine_names, df=df, k=k, yaw_angles=yaw)

    filtered_modelled = run(filtered_df)
    results: dict = {"k": k, "turbines": {}}
    for wtg, role in [
        (PREDICTION_WTG, "prediction"),
        (CALIBRATION_WTG, "calibration"),
        (FREE_STREAM_WTG, "diagnostic"),
    ]:
        results["turbines"][wtg] = {
            "role": role,
            **error_metrics(measured=filtered_df[f"{wtg}_power"], modelled=filtered_modelled[wtg]),
        }
        logger.info("%s (%s): %s", wtg, role, results["turbines"][wtg])
    results_path = out_dir / "results.json"
    logger.info("Writing: %s", results_path)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # minutes without a misalignment fall back to zero yaw so the whole window is plotted
    plot_df = minute_df.loc[minute_df[FREE_STREAM_COLS].notna().all(axis=1)].copy()
    plot_df[f"{STEERING_WTG}_yaw_misalignment"] = plot_df[f"{STEERING_WTG}_yaw_misalignment"].fillna(0)
    plot_time_series(
        minute_df=minute_df,
        modelled_df=run(plot_df),
        filtered_df=filtered_df,
        k=k,
        out_dir=out_dir,
    )
    for wtg, label in [(CALIBRATION_WTG, "calibration"), (PREDICTION_WTG, "prediction")]:
        plot_scatter(
            measured=filtered_df[f"{wtg}_power"],
            modelled=filtered_modelled[wtg],
            title=f"{wtg} ({label}), toggle-off minutes, k={k:.3f}",
            plot_path=out_dir / f"scatter_{wtg}.png",
        )


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    setup_logger(out_dir / f"{Path(__file__).stem}.log")
    main(out_dir)
