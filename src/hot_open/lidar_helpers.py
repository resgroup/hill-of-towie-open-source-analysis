"""Helpers for loading and processing 10-min and fastlog data from the ZX300 and ZXTM LiDARs."""

import logging
from pathlib import Path
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from wind_up.reanalysis_data import MastOrLiDARDataset

from hot_open.fastlog_helpers import _generate_dates_in_range
from hot_open.settings import get_cache_dir, get_data_dir
from hot_open.sourcing_data import ensure_extracted

logger = logging.getLogger(__name__)


def load_zx_lidar_10min_data(  # noqa: C901
    *,
    data_dir: Path,
    lidar_unit_id: str,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    remove_bad_values: bool = False,
) -> pd.DataFrame:
    """Load 10-min ZX300/ZXTM LiDAR parquet files for a single unit and date range."""
    ensure_extracted("lidar_data.zip", data_dir=get_data_dir())
    dfs = []
    for fname in (data_dir / "timeseries" / lidar_unit_id).glob("*.parquet"):
        if not fname.stem.startswith("Wind10"):
            continue
        expected_date = pd.to_datetime(str(fname.stem).split("@")[-1].split(".")[0], format="Y%Y_M%m_D%d", utc=True)
        if expected_date < (start_dt - pd.Timedelta(days=2)) or expected_date > (end_dt_excl + pd.Timedelta(days=1)):
            continue
        logger.info("Reading: %s", fname)
        _df = pd.read_parquet(fname)
        _df = _df.drop(columns=[x for x in _df.columns if x.startswith("Checksum")])
        # find a good timestamp column
        if "Timestamp (ISO 8601)" in _df.columns:
            _df["timestamp"] = pd.to_datetime(_df["Timestamp (ISO 8601)"], utc=True)
            _df["Timestamp (ISO 8601)"] = _df["Timestamp (ISO 8601)"].astype(
                str
            )  # ensure this guy is a string to avoid mixed types
        elif "Time and Date" in _df.columns:
            _df["timestamp"] = pd.to_datetime(_df["Time and Date"], format="%m/%d/%Y %I:%M:%S %p", utc=True)
            _df["Time and Date"] = _df["Time and Date"].astype(str)  # ensure this guys is a string to avoid mixed types
        if not is_datetime64_any_dtype(_df["timestamp"]):
            ts_dtype = _df["timestamp"].dtype
            msg = f"{ts_dtype=}"
            raise ValueError(msg)
        if not (_df["timestamp"] - expected_date).dt.total_seconds().abs().max() < (24 * 3600):
            ts_min = _df["timestamp"].min()
            ts_max = _df["timestamp"].max()
            msg = f"something is wrong:\n{fname.stem=}\n{expected_date=}\n{ts_min=}\n{ts_max=}"
            raise ValueError(msg)
        dfs.append(_df)
    return_df = pd.concat(dfs).set_index("timestamp", drop=True).sort_index() if dfs else pd.DataFrame()
    if return_df.empty:
        return return_df
    if remove_bad_values:
        for zx_bad_value in [9999, 9991]:
            return_df = return_df.replace(zx_bad_value, np.nan)
    return return_df[(return_df.index >= start_dt) & (return_df.index < end_dt_excl)]


def load_hot_lidar_10min_data(
    *,
    data_dir: Path,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
) -> list[MastOrLiDARDataset]:
    """Load HoT ZX300 Lidar data."""
    lidar_datasets = []

    lidar_unit_id = "2428"
    lidar_model = "ZX300"
    lidar_datasets.append(
        MastOrLiDARDataset(
            id=f"{lidar_model}_{lidar_unit_id}",
            data=load_zx_lidar_10min_data(
                data_dir=data_dir, lidar_unit_id=lidar_unit_id, start_dt=start_dt, end_dt_excl=end_dt_excl
            ),
        )
    )

    lidar_unit_id = "5060"
    lidar_model = "ZTM"
    df_5060 = load_zx_lidar_10min_data(
        data_dir=data_dir, lidar_unit_id=lidar_unit_id, start_dt=start_dt, end_dt_excl=end_dt_excl
    )
    df_5060["Met Compass Bearing (deg)"] = (
        df_5060["Met Compass Bearing (deg)"] - 131.5
    ) % 360  # TODO(@aclerc): fix in wind-up config instead  # noqa: FIX002, TD003
    lidar_datasets.append(
        MastOrLiDARDataset(
            id=f"{lidar_model}_{lidar_unit_id}",
            data=df_5060,
        )
    )

    return lidar_datasets


@overload
def calc_air_density_iec(*, temp_c: float, pressure_mbar: float, humidity_percent: float) -> float: ...


@overload
def calc_air_density_iec(*, temp_c: pd.Series, pressure_mbar: pd.Series, humidity_percent: pd.Series) -> pd.Series: ...


def calc_air_density_iec(
    *, temp_c: float | pd.Series, pressure_mbar: float | pd.Series, humidity_percent: float | pd.Series
) -> Any:
    """Calculate air density as per IEC 61400-12-1.

    Parameters
    ----------
    temp_c : pd.Series
        Air temperature in degrees Celsius
    pressure_mbar : pd.Series
        Atmospheric pressure in millibars (mbar) or hectopascals (hPa)
    humidity_percent : pd.Series
        Relative humidity as a percentage (0-100)

    Returns
    -------
    pd.Series
        Air density in kg/m³

    """
    temp_k = temp_c + 273.15
    pressure_pa = pressure_mbar * 100
    r0 = 287.05  # gas constant of dry air
    rw = 461.5  # gas constant of water vapour
    vapour_pressure_pa = 0.0000205 * np.exp(0.0631846 * temp_k)
    return (1 / temp_k) * (pressure_pa / r0 - (humidity_percent / 100) * vapour_pressure_pa * (1 / r0 - 1 / rw))


ZX_ZTM_DEVICE_ID_MIN = 5000
MIN_PRESSURE_MBAR = 700
MAX_PRESSURE_MBAR = 1200
LIDAR_AIR_DENSITY_COL = "Air Density (kg/m3)"
MIN_AIR_DENSITY_KGPM3 = 0.9
MAX_AIR_DENSITY_KGPM3 = 1.4


def load_zx_lidar_fl_data(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    data_dir: Path,
    lidar_unit_id: str,
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    remove_bad_values: bool = False,
    add_air_density: bool = False,
) -> pd.DataFrame:
    """Return ZX LiDAR fastlog dataframe for any wind farm."""
    cache_fname = f"{lidar_unit_id}_{start_dt.strftime('%Y%m%d%H%M%S')}_{end_dt_excl.strftime('%Y%m%d%H%M%S')}.parquet"
    cache_dir = get_cache_dir()
    if cache_dir is not None and (cache_dir / "load_zx_lidar_fl_data" / cache_fname).exists():
        cache_path = cache_dir / "load_zx_lidar_fl_data" / cache_fname
        logger.info("Reading: %s", cache_path)
        return pd.read_parquet(cache_path)
    ensure_extracted("lidar_data.zip", data_dir=get_data_dir())
    device_decr = "ZTM" if int(lidar_unit_id) >= ZX_ZTM_DEVICE_ID_MIN else ""
    file_paths = [
        data_dir
        / "timeseries"
        / str(lidar_unit_id)
        / f"Wind_{device_decr}{lidar_unit_id}@{d.strftime('Y%Y_M%m_D%d')}.parquet"
        for d in _generate_dates_in_range(start_dt=start_dt, end_dt_excl=end_dt_excl)
    ]
    dfs = []
    for file_path in file_paths:
        try:
            logger.info("Reading: %s", file_path)
            _df = pd.read_parquet(file_path)
        except FileNotFoundError:
            msg = f"File {file_path} not found."
            logger.warning(msg)
            continue
        _df = _df.drop(columns=[x for x in _df.columns if x.startswith("Checksum")])
        # find a good timestamp column
        if "Timestamp (ISO 8601)" in _df.columns:
            _df["timestamp"] = pd.to_datetime(_df["Timestamp (ISO 8601)"])
            _df["Timestamp (ISO 8601)"] = _df["Timestamp (ISO 8601)"].astype(
                str
            )  # ensure this guys is a string to avoid mixed types
        elif "Time and Date" in _df.columns:
            _df["timestamp"] = pd.to_datetime(_df["Time and Date"], format="%m/%d/%Y %I:%M:%S %p")
            _df["Time and Date"] = _df["Time and Date"].astype(str)  # ensure this guys is a string to avoid mixed types
        if not is_datetime64_any_dtype(_df["timestamp"]):
            msg = f"{_df['timestamp'].dtype=}"
            raise ValueError(msg)
        expected_date = pd.to_datetime(str(file_path).split("@")[-1].split(".")[0], format="Y%Y_M%m_D%d")
        if not (_df["timestamp"] - expected_date).dt.total_seconds().abs().max() < (24 * 3600):
            msg = (
                f"something is wrong:\n{file_path=}\n{expected_date=}"
                f"\n{_df['timestamp'].min()=}\n{_df['timestamp'].max()=}"
            )
            raise ValueError(msg)
        dfs.append(_df)
    return_df = pd.concat(dfs).set_index("timestamp", drop=True) if dfs else pd.DataFrame()
    if return_df.empty:
        return return_df
    return_df = return_df[~return_df.index.duplicated(keep="last")].sort_index()
    if remove_bad_values:
        # replace bad values with NaN
        for zx_bad_value in [9999, 9991]:
            return_df = return_df.replace(zx_bad_value, np.nan)
        # replace bad Met Pressure values with NaN
        bad_pressure = (return_df["Met Pressure (mbar)"] < MIN_PRESSURE_MBAR) | (
            return_df["Met Pressure (mbar)"] > MAX_PRESSURE_MBAR
        )
        return_df.loc[bad_pressure, "Met Pressure (mbar)"] = np.nan
    if add_air_density:
        # add air density
        return_df[LIDAR_AIR_DENSITY_COL] = calc_air_density_iec(
            temp_c=return_df["Met Air Temp. (C)"],
            pressure_mbar=return_df["Met Pressure (mbar)"],
            humidity_percent=return_df["Met Humidity (%)"],
        )
        if remove_bad_values:
            bad_air_density = (return_df[LIDAR_AIR_DENSITY_COL] < MIN_AIR_DENSITY_KGPM3) | (
                return_df[LIDAR_AIR_DENSITY_COL] > MAX_AIR_DENSITY_KGPM3
            )
            return_df.loc[bad_air_density, LIDAR_AIR_DENSITY_COL] = np.nan
    if return_df.index.tzinfo is None:  # type: ignore[attr-defined]
        # LiDAR data is in UTC
        return_df.index = pd.to_datetime(return_df.index, utc=True)
    return_df = (
        return_df[(return_df.index >= pd.Timestamp(start_dt)) & (return_df.index < pd.Timestamp(end_dt_excl))]
    ).sort_index()
    if cache_dir is not None:
        (cache_dir / "lidar_raw").mkdir(exist_ok=True, parents=True)
        raw_cache_path = cache_dir / "lidar_raw" / cache_fname
        logger.info("Writing: %s", raw_cache_path)
        return_df.to_parquet(raw_cache_path)
        logger.info("Reading: %s", raw_cache_path)
        return pd.read_parquet(raw_cache_path)
    return return_df


_MIN_FIT_POINTS = 2


def extract_data(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Return a sub-DataFrame of the columns whose name starts with `prefix`.

    Column names are replaced with the height in metres parsed from the original
    name (e.g. "Horizontal Wind Speed (m/s) at 58m" -> 58.0), and the result is
    sorted by ascending height.
    """
    cols = [str(col) for col in df.columns if str(col).startswith(prefix)]
    heights = np.array([float(col.split("at ")[1].split("m")[0]) for col in cols])
    out = df[cols].copy()
    out.columns = heights
    return out.sort_index(axis=1)


def add_shear_and_veer(df: pd.DataFrame) -> pd.DataFrame:
    """Append shear and veer columns to a LiDAR DataFrame.

    Adds: "Vertical Wind Shear Exponent", "Normalised Mean Shear Fit Residuals",
    "Vertical Wind Veer", "Vertical Wind Veer R Squared".
    """
    ws_prefix = "Horizontal Wind Speed (m/s)"
    wd_prefix = "Wind Direction (deg)"

    ws = extract_data(df, ws_prefix)
    wd = extract_data(df, wd_prefix)
    df["Vertical Wind Shear Exponent"], df["Normalised Mean Shear Fit Residuals"] = calculate_shear(ws)
    df["Vertical Wind Veer"], df["Vertical Wind Veer R Squared"] = calculate_veer(wd)
    return df


def calculate_shear(ws: pd.DataFrame) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Fit a power-law wind shear exponent (alpha) per row.

    Returns the per-row alpha and the RMS of log-space fit residuals.
    """
    n_times, _ = ws.shape
    alphas = np.full(n_times, np.nan, dtype=float)
    residuals = np.full(n_times, np.nan, dtype=float)

    values = ws.to_numpy()
    heights = ws.columns.to_numpy()
    order = np.argsort(heights)
    heights = heights[order]
    values = values[:, order]
    for i in range(n_times):
        row = values[i, :]
        valid = np.isfinite(row) & (row > 0.0)
        valid_idx = np.where(valid)[0]
        if valid_idx.size < _MIN_FIT_POINTS:
            continue
        valid_heights = heights[valid_idx]
        valid_ws = row[valid_idx]

        x = np.log(valid_heights)
        y = np.log(valid_ws)

        coeffs = np.polyfit(x, y, 1)
        alpha_i, c_i = coeffs
        yhat = alpha_i * x + c_i
        resid_i = np.sqrt(np.mean((y - yhat) ** 2))

        alphas[i] = alpha_i
        residuals[i] = resid_i
    return alphas, residuals


def calculate_veer(wd: pd.DataFrame) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Fit a linear veer (deg/m) per row using a circular-mean correction.

    Returns the per-row slope and the R^2 of the linear fit.
    """
    n_times, _ = wd.shape
    m = np.full(n_times, np.nan, dtype=float)
    r2 = np.full(n_times, np.nan, dtype=float)

    values = wd.to_numpy()
    heights = wd.columns.to_numpy()
    for i in range(n_times):
        row = values[i, :]
        valid = np.isfinite(row) & (row > 0.0)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) < _MIN_FIT_POINTS:
            continue
        valid_heights = heights[valid_idx]
        valid_dir = row[valid_idx]

        sine_mean = np.mean(np.sin(np.radians(valid_dir)))
        cosine_mean = np.mean(np.cos(np.radians(valid_dir)))
        dir_mean = np.degrees(np.arctan2(sine_mean, cosine_mean))
        corrected_valid_directions = (((valid_dir - dir_mean) + 180.0) % 360.0) - 180.0

        coeffs = np.polyfit(valid_heights, corrected_valid_directions, 1)
        m_i, _c_i = coeffs
        p = np.poly1d(coeffs)
        yhat = p(valid_heights)
        ybar = np.sum(corrected_valid_directions) / len(corrected_valid_directions)
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((corrected_valid_directions - ybar) ** 2)
        r2_i = ssreg / sstot if sstot != 0 else np.nan
        m[i] = m_i
        r2[i] = r2_i
    return m, r2
