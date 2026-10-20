"""FLORIS toggle-off baseline: predict T07 power from T01 free-stream inputs, calibrated to unwaked T03.

The model is built from public information only: Zenodo turbine metadata, a generic PyWake turbine
(see ``make_turbine_yaml.py``) and FLORIS default wake parameters (``floris_inputs/hot_public.yaml``).

Per minute, the free stream is ``k`` times T01's nacelle anemometer wind speed, T01's DY consensus
wind direction and a TI from T01's 1 Hz anemometer. ``k`` is fitted so FLORIS reproduces T03 power,
which is unwaked in the T03 -> T07 sector; T07 power is then an out-of-sample prediction.

Run with ``uv sync --group wedowind`` first.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from floris import FlorisModel, TimeSeries
from geographiclib.geodesic import Geodesic
from scipy.optimize import minimize_scalar

from hot_open.circular_math import circ_mean_resample_degrees
from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.settings import get_filestore_dir, get_out_dir
from hot_open.unpack import unpack_local_meta_data
from scripts.logger import setup_logger

logger = logging.getLogger(__name__)

FLORIS_INPUTS_DIR = Path(__file__).parent / "floris_inputs"
FARM_YAML_PATH = FLORIS_INPUTS_DIR / "hot_public.yaml"

FREE_STREAM_WTG = "T01"
CALIBRATION_WTG = "T03"
PREDICTION_WTG = "T07"
WTGS = [FREE_STREAM_WTG, CALIBRATION_WTG, PREDICTION_WTG]

LOAD_START = pd.Timestamp("2026-02-20", tz="UTC")
LOAD_END_EXCL = pd.Timestamp("2026-02-23", tz="UTC")
# slide 12 of the WindEurope Tech 2026 presentation
SLIDE_WINDOW_START = pd.Timestamp("2026-02-20 00:00", tz="UTC")
SLIDE_WINDOW_END_EXCL = pd.Timestamp("2026-02-20 04:20", tz="UTC")

POWER_COL = "ActPower_Value"
WS_COL = "AcWindSp_AcWindSp"
GEN_STATE_COL = "GenState_GenState"
ACT_LIMIT_COL = "ActLimit_Power"
TOGGLE_COL = "computed_driver_post_processed_toggle_state"
WD_COL = "computed_core_post_processed_consensus_wind_direction_true_degrees"

ON_GRID_GEN_STATE = "LargeOnGrd"
RATED_POWER_KW = 2300.0
SECONDS_PER_MINUTE = 60
STEERING_SECTOR_DEG = (192.0, 224.0)
TI_CLIP = (0.03, 0.25)
WIND_SHEAR = 0.12
K_BOUNDS = (0.7, 1.3)


def local_east_north(
    *, latitudes: npt.ArrayLike, longitudes: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return east and north coordinates in metres, with the minimum easting and northing at 0.

    Points are placed by their WGS84 geodesic distance and bearing from the centroid of all points
    (an azimuthal equidistant projection), so north is true north.
    """
    # Vendored from wind-up v1 (wind_up.geodesy.local_east_north); import it once wind-up v1 is released.
    lats = np.asarray(latitudes, dtype=float)
    lons = np.asarray(longitudes, dtype=float)
    centroid = (float(lats.mean()), float(lons.mean()))
    east = np.empty(len(lats))
    north = np.empty(len(lats))
    for i, (lat, lon) in enumerate(zip(lats, lons, strict=True)):
        result = Geodesic.WGS84.Inverse(centroid[0], centroid[1], lat, lon)
        east[i] = result["s12"] * np.sin(np.radians(result["azi1"]))
        north[i] = result["s12"] * np.cos(np.radians(result["azi1"]))
    return east - east.min(), north - north.min()


def load_hot_floris_model(metadata_df: pd.DataFrame) -> tuple[FlorisModel, list[str]]:
    """Return the public FLORIS model with its layout set from ``metadata_df`` and its turbine names.

    ``metadata_df`` has ``Name``, ``Latitude`` and ``Longitude`` columns, as from ``unpack_local_meta_data``.
    """
    x, y = local_east_north(latitudes=metadata_df["Latitude"], longitudes=metadata_df["Longitude"])
    fmodel = FlorisModel(FARM_YAML_PATH)
    fmodel.set(layout_x=x, layout_y=y, wind_shear=WIND_SHEAR)
    return fmodel, metadata_df["Name"].tolist()


def run_floris(
    fmodel: FlorisModel,
    *,
    turbine_names: list[str],
    wind_speeds: npt.ArrayLike,
    wind_directions: npt.ArrayLike,
    turbulence_intensities: npt.ArrayLike,
    index: pd.Index | None = None,
) -> pd.DataFrame:
    """Run FLORIS with zero yaw misalignment and return power in kW, one column per turbine."""
    time_series = TimeSeries(
        wind_directions=np.asarray(wind_directions, dtype=float),
        wind_speeds=np.asarray(wind_speeds, dtype=float),
        turbulence_intensities=np.asarray(turbulence_intensities, dtype=float),
    )
    fmodel.set(wind_data=time_series, yaw_angles=np.zeros((time_series.n_findex, len(turbine_names))))
    fmodel.run()
    return pd.DataFrame(fmodel.get_turbine_powers() / 1000, index=index, columns=turbine_names)


def fit_ws_scale(
    *,
    modelled_power_fn: Callable[[float], npt.NDArray[np.float64]],
    measured_power: npt.ArrayLike,
    bounds: tuple[float, float] = K_BOUNDS,
) -> float:
    """Return the wind speed scale ``k`` minimising the RMSE of ``modelled_power_fn(k)`` against ``measured_power``."""
    measured = np.asarray(measured_power, dtype=float)

    def rmse(k: float) -> float:
        return float(np.sqrt(np.mean((modelled_power_fn(k) - measured) ** 2)))

    result = minimize_scalar(rmse, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    k = float(result.x)
    if min(abs(k - bounds[0]), abs(k - bounds[1])) < 1e-3 * (bounds[1] - bounds[0]):
        logger.warning("fitted k=%.4f is on a search bound %s", k, bounds)
    return k


def check_data_present(fl_df: pd.DataFrame, *, wtgs: list[str]) -> None:
    present = set(fl_df.columns.get_level_values(0))
    for wtg in wtgs:
        for day in pd.date_range(LOAD_START, LOAD_END_EXCL, freq="D", inclusive="left"):
            day_power = (
                fl_df.loc[(fl_df.index >= day) & (fl_df.index < day + pd.Timedelta(days=1)), (wtg, POWER_COL)]
                if wtg in present
                else pd.Series(dtype=float)
            )
            if day_power.notna().sum() == 0:
                msg = f"no fastlog power data for {wtg} on {day.date()}"
                raise ValueError(msg)


def make_minute_df(fl_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the 1 Hz fastlog to one row per minute with free-stream inputs and per-turbine power and flags."""
    one_min = pd.Timedelta(minutes=1)
    fs = fl_df[FREE_STREAM_WTG]
    ws_resampler = fs[WS_COL].resample(one_min)
    minute_df = pd.DataFrame(
        {
            "ws_raw": ws_resampler.mean(),
            "ws_std": ws_resampler.std(),
            "wd": circ_mean_resample_degrees(fs[[WD_COL]], resample_timedelta=one_min)[WD_COL],
        }
    )
    minute_df["ti"] = (minute_df["ws_std"] / minute_df["ws_raw"]).clip(*TI_CLIP)
    for wtg in WTGS:
        d = fl_df[wtg]
        normal = (d[POWER_COL] > 0) & d[GEN_STATE_COL].eq(ON_GRID_GEN_STATE) & (d[ACT_LIMIT_COL] >= RATED_POWER_KW)
        toggle = d[TOGGLE_COL].resample(one_min)
        minute_df[f"{wtg}_power"] = d[POWER_COL].resample(one_min).mean()
        minute_df[f"{wtg}_normal"] = normal.resample(one_min).sum() == SECONDS_PER_MINUTE
        minute_df[f"{wtg}_toggle_off"] = (toggle.max() == 0) & (toggle.count() == SECONDS_PER_MINUTE)
        minute_df[f"{wtg}_toggle_mean"] = toggle.mean()
    return minute_df


def filter_minutes(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Keep toggle-off, in-sector minutes with all turbines operating normally."""
    lo, hi = STEERING_SECTOR_DEG
    masks = {
        "free-stream inputs present": minute_df[["ws_raw", "wd", "ti"]].notna().all(axis=1),
        "toggle off all minute": minute_df[[f"{wtg}_toggle_off" for wtg in WTGS]].all(axis=1),
        f"direction in {lo:g}-{hi:g} deg": minute_df["wd"].between(lo, hi),
        "all turbines normal": minute_df[[f"{wtg}_normal" for wtg in WTGS]].all(axis=1),
    }
    keep = pd.Series(data=True, index=minute_df.index)
    removed = {}
    for name, mask in masks.items():
        removed[name] = int((keep & ~mask).sum())
        keep &= mask
    logger.info("minutes removed by each filter step: %s; %d kept", removed, keep.sum())
    if not keep.any():
        msg = f"no minutes survive the filter; removed by each step: {removed}"
        raise ValueError(msg)
    return minute_df.loc[keep]


def error_metrics(*, measured: pd.Series, modelled: pd.Series) -> dict[str, float | int]:
    err = modelled - measured
    return {"rmse_kw": float(np.sqrt((err**2).mean())), "bias_kw": float(err.mean()), "count": len(err)}


def plot_time_series(*, minute_df: pd.DataFrame, modelled_df: pd.DataFrame, k: float, out_dir: Path) -> None:
    in_window = (minute_df.index >= SLIDE_WINDOW_START) & (minute_df.index < SLIDE_WINDOW_END_EXCL)
    window_df = minute_df.loc[in_window]
    window_modelled = modelled_df.reindex(window_df.index)
    toggle_on = window_df[f"{CALIBRATION_WTG}_toggle_mean"] > 0.5

    fig, axes = plt.subplots(len(WTGS), 1, figsize=(14, 10), sharex=True)
    for ax, wtg in zip(axes, WTGS, strict=True):
        ax.fill_between(
            window_df.index,
            0,
            1,
            where=toggle_on,
            step="post",
            transform=ax.get_xaxis_transform(),
            color="tab:green",
            alpha=0.15,
            label="toggle on",
        )
        ax.plot(window_df.index, window_df[f"{wtg}_power"], drawstyle="steps-post", label="measured")
        ax.plot(window_modelled.index, window_modelled[wtg], drawstyle="steps-post", linestyle="--", label="FLORIS")
        ax.set_ylabel(f"{wtg} power [kW]")
        ax.grid(visible=True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    axes[-1].set_xlabel("timestamp (UTC)")
    fig.suptitle(
        f"Measured vs FLORIS power, no yaw misalignment in the model, k={k:.3f}\n"
        f"{SLIDE_WINDOW_START:%Y-%m-%d %H:%M} to {SLIDE_WINDOW_END_EXCL:%H:%M} UTC"
    )
    plot_path = out_dir / "time_series_slide_window.png"
    logger.info("Writing: %s", plot_path)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(*, measured: pd.Series, modelled: pd.Series, title: str, plot_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(measured, modelled, s=8, alpha=0.5)
    ax.plot([0, RATED_POWER_KW], [0, RATED_POWER_KW], color="black", linewidth=0.8)
    ax.set_xlabel("measured power [kW]")
    ax.set_ylabel("FLORIS power [kW]")
    ax.set_xlim(0, RATED_POWER_KW * 1.05)
    ax.set_ylim(0, RATED_POWER_KW * 1.05)
    ax.set_aspect("equal")
    ax.grid(visible=True, alpha=0.3)
    ax.set_title(title)
    logger.info("Writing: %s", plot_path)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(out_dir: Path) -> None:
    fl_df = load_hot_fl_data(
        data_dir=get_filestore_dir(),
        wtg_numbers=[int(wtg[1:]) for wtg in WTGS],
        start_dt=LOAD_START,
        end_dt_excl=LOAD_END_EXCL,
        extra_tags=[TOGGLE_COL, WD_COL],
    )
    check_data_present(fl_df, wtgs=WTGS)
    minute_df = make_minute_df(fl_df)
    filtered_df = filter_minutes(minute_df)

    fmodel, turbine_names = load_hot_floris_model(unpack_local_meta_data())

    def modelled_df_for(df: pd.DataFrame, k: float) -> pd.DataFrame:
        return run_floris(
            fmodel,
            turbine_names=turbine_names,
            wind_speeds=k * df["ws_raw"].clip(lower=0),
            wind_directions=df["wd"],
            turbulence_intensities=df["ti"],
            index=df.index,
        )

    k = fit_ws_scale(
        modelled_power_fn=lambda k: modelled_df_for(filtered_df, k)[CALIBRATION_WTG].to_numpy(),
        measured_power=filtered_df[f"{CALIBRATION_WTG}_power"],
    )
    logger.info("fitted k=%.4f on %d %s minutes", k, len(filtered_df), CALIBRATION_WTG)

    filtered_modelled = modelled_df_for(filtered_df, k)
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

    inputs_present = minute_df[["ws_raw", "wd", "ti"]].notna().all(axis=1)
    plot_time_series(
        minute_df=minute_df,
        modelled_df=modelled_df_for(minute_df.loc[inputs_present], k),
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
