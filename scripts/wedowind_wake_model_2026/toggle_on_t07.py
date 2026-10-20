"""FLORIS with T03's measured yaw: predict T03 and T07 power on toggle-on (wake steering) minutes.

The model, free stream and ``k`` (fitted on toggle-off T03 minutes) are as in ``toggle_off_t07.py``; see
``hot_floris.py``. T03 gets its measured yaw misalignment (yaw position minus its own DY consensus wind
direction) on toggle-off and toggle-on minutes alike, and every other turbine stays at zero yaw. FLORIS is also
run with zero yaw on the toggle-on minutes to show the modelled steering effect.

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
    PREDICTION_WTG,
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

POWER_WTGS = [CALIBRATION_WTG, PREDICTION_WTG]


def plot_time_series(
    *,
    minute_df: pd.DataFrame,
    yawed_df: pd.DataFrame,
    unyawed_df: pd.DataFrame,
    on_df: pd.DataFrame,
    k: float,
    out_dir: Path,
) -> None:
    """Plot T03's misalignment, then measured and FLORIS (with and without T03 yaw) T03 and T07 power."""
    yawed_df = yawed_df.reindex(minute_df.index)
    unyawed_df = unyawed_df.reindex(minute_df.index)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    ax_yaw, *power_axes = axes
    for ax in axes:
        shade_toggle_on(ax, minute_df=minute_df)

    ax_yaw.plot(
        minute_df.index,
        minute_df[f"{STEERING_WTG}_yaw_misalignment"],
        drawstyle="steps-post",
        label=f"{STEERING_WTG} yaw position - {STEERING_WTG} DY consensus (centred 5 min)",
    )
    ax_yaw.axhline(0, color="black", linewidth=0.8)
    ax_yaw.set_ylabel(f"{STEERING_WTG} yaw misalignment [deg]")

    for ax, wtg in zip(power_axes, POWER_WTGS, strict=True):
        ax.plot(minute_df.index, minute_df[f"{wtg}_power"], drawstyle="steps-post", label="measured")
        for modelled_df, label in [(yawed_df, f"{STEERING_WTG} measured yaw"), (unyawed_df, "zero yaw")]:
            bias = round((modelled_df.loc[on_df.index, wtg] - on_df[f"{wtg}_power"]).mean())
            ax.plot(
                modelled_df.index,
                modelled_df[wtg],
                drawstyle="steps-post",
                linestyle="--",
                label=f"FLORIS, {label} (toggle-on bias {bias:+d} kW)",
            )
        ax.set_ylabel(f"{wtg} power [kW]")

    for ax in axes:
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    axes[-1].set_xlabel("timestamp (UTC)")
    fig.suptitle(
        f"FLORIS with {STEERING_WTG}'s measured yaw, k={k:.3f} from toggle-off {CALIBRATION_WTG}, "
        f"{len(on_df)} toggle-on minutes\n{WINDOW_START:%Y-%m-%d %H:%M} to {WINDOW_END_EXCL:%H:%M} UTC"
    )
    plot_path = out_dir / "time_series_slide_window.png"
    logger.info("Writing: %s", plot_path)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    minute_df = load_minute_df()
    fmodel, turbine_names = load_hot_floris_model(unpack_local_meta_data())
    k = fit_toggle_off_k(minute_df=minute_df, fmodel=fmodel, turbine_names=turbine_names)
    off_df = filter_minutes(minute_df, toggle="off")
    on_df = filter_minutes(minute_df, toggle="on")

    def run(df: pd.DataFrame, *, yawed: bool) -> pd.DataFrame:
        yaw = yaw_angles_for(df=df, turbine_names=turbine_names, wtg=STEERING_WTG) if yawed else None
        return run_floris_minutes(fmodel, turbine_names=turbine_names, df=df, k=k, yaw_angles=yaw)

    off_modelled = run(off_df, yawed=True)
    on_yawed = run(on_df, yawed=True)
    on_unyawed = run(on_df, yawed=False)

    results: dict = {
        "k": k,
        "mean_yaw_misalignment_deg": {
            "toggle_off": float(off_df[f"{STEERING_WTG}_yaw_misalignment"].mean()),
            "toggle_on": float(on_df[f"{STEERING_WTG}_yaw_misalignment"].mean()),
        },
        "turbines": {},
    }
    for wtg in POWER_WTGS:
        toggle_off = error_metrics(measured=off_df[f"{wtg}_power"], modelled=off_modelled[wtg])
        toggle_on = error_metrics(measured=on_df[f"{wtg}_power"], modelled=on_yawed[wtg])
        results["turbines"][wtg] = {
            "toggle_off_measured_yaw": toggle_off,
            "toggle_on_measured_yaw": toggle_on,
            "modelled_steering_effect_kw": float((on_yawed[wtg] - on_unyawed[wtg]).mean()),
            "bias_shift_kw": toggle_on["bias_kw"] - toggle_off["bias_kw"],
        }
        logger.info("%s: %s", wtg, results["turbines"][wtg])
    results_path = out_dir / "results.json"
    logger.info("Writing: %s", results_path)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # minutes without a misalignment fall back to zero yaw so the whole window is plotted
    plot_df = minute_df.loc[minute_df[FREE_STREAM_COLS].notna().all(axis=1)].copy()
    plot_df[f"{STEERING_WTG}_yaw_misalignment"] = plot_df[f"{STEERING_WTG}_yaw_misalignment"].fillna(0)
    plot_time_series(
        minute_df=minute_df,
        yawed_df=run(plot_df, yawed=True),
        unyawed_df=run(plot_df, yawed=False),
        on_df=on_df,
        k=k,
        out_dir=out_dir,
    )
    for wtg in POWER_WTGS:
        plot_scatter(
            measured=on_df[f"{wtg}_power"],
            modelled=on_yawed[wtg],
            title=f"{wtg}, toggle-on minutes, FLORIS with {STEERING_WTG} measured yaw, k={k:.3f}",
            plot_path=out_dir / f"scatter_{wtg}.png",
        )


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    setup_logger(out_dir / f"{Path(__file__).stem}.log")
    main(out_dir)
