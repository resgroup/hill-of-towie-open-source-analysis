"""Helpers for loading and resampling Siemens fastlog (FL) data from the Filestore directory tree."""

import base64
import datetime as dt
import hashlib
import inspect
import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid

from hot_open.circular_math import circ_mean_resample_degrees
from hot_open.settings import get_cache_dir, get_data_dir, get_filestore_dir
from hot_open.sourcing_data import ensure_extracted

logger = logging.getLogger(__name__)

TIMESTAMP_NAME = "timestamp"
SIEMENS_TAGS = [
    "AcWindDr_Source",
    "AcWindDr_Value",
    "AcWindSp_AcWindSp",
    "ActLimit_Power",
    "ActPower_Value",
    "GenState_GenState",
    "MainSRpm_Value",
    "PitcPosA_Value",
    "PitcPosB_Value",
    "PitcPosC_Value",
    "PowerRed_PowerRed",
    "PowerRef_PowerRef",
    "ReactPwr_Value",
    "YawExec_YawExec",
    "YawPos_Value",
]


SIEMENS_PARKS = {"HOT"}


def load_hot_fl_data(  # noqa: PLR0913
    *,
    data_dir: Path,
    wtg_numbers: Sequence[int],
    start_dt: pd.Timestamp,
    end_dt_excl: pd.Timestamp,
    use_turbine_names: bool = True,  # if False serial numbers are used to identify turbines
    extra_tags: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Load HOT fastlog data for specified turbines and time range.

    Returns a wide-format DataFrame with a MultiIndex column (turbine_name, tag).
    """
    # Check the actual filesystem rather than env-var presence: a user can set
    # HOT_OPEN_FILESTORE_DIR to point at a populated local copy, or leave it
    # unset and let the Zenodo zip extract into the default location.
    filestore_dir = get_filestore_dir()
    if not (filestore_dir / "FL").exists():
        if os.getenv("HOT_OPEN_FILESTORE_DIR") is None:
            ensure_extracted("turbine_fastlog.zip", data_dir=get_data_dir())
        else:
            msg = (
                f"HOT_OPEN_FILESTORE_DIR={filestore_dir} but it does not contain an 'FL' subdirectory. "
                "Populate it with the Hill of Towie fastlog tree (FL/HOT/<device_id>/<date>/...), "
                "or unset HOT_OPEN_FILESTORE_DIR to auto-download from Zenodo."
            )
            raise FileNotFoundError(msg)
    park_id = "HOT"
    tags = [*_get_tag_list_from_park_id(park_id), *extra_tags] if extra_tags is not None else None
    fl_df = get_fl_resampled(
        timebase_s=1,
        park_id=park_id,
        device_ids=[str(x + 2304509) for x in wtg_numbers],
        tags=tags,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        cache_dir=get_cache_dir(),
        filestore_dir=data_dir,
    )
    if use_turbine_names:
        cols = fl_df.columns
        fl_df.columns = cols.set_levels(  # type:ignore[attr-defined]
            [{x: f"T{int(x) - 2304509:02d}" for x in cols.get_level_values(0).unique()}[x] for x in cols.levels[0]],  # type:ignore[attr-defined]
            level=0,
        )
    return fl_df


def get_fl_resampled(  # noqa: PLR0913
    *,
    park_id: str,
    device_ids: list[str],
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    timebase_s: int = 1,
    tags: Sequence[str] | None = None,
    busy_tags: Sequence[str] | None = None,
    minmax_tags: Sequence[str] | None = None,
    min_data_count: float | None = None,
    cache_dir: Path | None = None,
    filestore_dir: Path | None = None,
) -> pd.DataFrame:
    """Return resampled fastlog data for multiple devices as a multi-level (device_id, tag) DataFrame."""
    filestore_dir = get_filestore_dir() if filestore_dir is None else filestore_dir
    device_id_dfs: dict[str, pd.DataFrame] = {}
    for device_id in device_ids:
        device_id_df = get_fl_resampled_one_device(
            park_id=park_id,
            device_id=device_id,
            start_dt=start_dt,
            end_dt_excl=end_dt_excl,
            timebase_s=timebase_s,
            filestore_dir=filestore_dir,
            tags=tags,
            busy_tags=busy_tags,
            minmax_tags=minmax_tags,
            min_data_count=min_data_count,
            cache_dir=cache_dir,
        )
        if not device_id_df.index.is_monotonic_increasing:
            msg = f"Resampled data index for {device_id} is not monotonic increasing."
            raise ValueError(msg)
        device_id_dfs[device_id] = device_id_df
    # Remove freq metadata which can confuse concat
    device_id_dfs_no_freq = {k: df.set_index(pd.DatetimeIndex(df.index, freq=None)) for k, df in device_id_dfs.items()}  # type: ignore[arg-type]
    resampled_df = pd.concat(device_id_dfs_no_freq, axis=1, names=["device_id", "tag"])
    if resampled_df.empty:
        return resampled_df
    resampled_df = resampled_df.resample(f"{timebase_s}s").last()

    return resampled_df.loc[
        (resampled_df.index >= resampled_df.dropna(how="all").index[0])
        & (resampled_df.index <= resampled_df.dropna(how="all").index[-1]),
        :,
    ]


def _get_fl_resampled_one_device_one_day(  # noqa: PLR0913
    *,
    park_id: str,
    device_id: str,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    timebase_s: int = 1,
    filestore_dir: Path | None = None,
    tags: Sequence[str] | None = None,
    busy_tags: Sequence[str] | None = None,
    minmax_tags: Sequence[str] | None = None,
    min_data_count: float | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    if (end_dt_excl - start_dt) > dt.timedelta(days=1):
        msg = f"Date range must be one day or less. Got {start_dt=} {end_dt_excl=}"
        raise ValueError(msg)
    if cache_dir is not None:
        frame = inspect.currentframe()
        if frame is None:
            msg = "Could not get current frame"
            raise RuntimeError(msg)
        args_info = inspect.getargvalues(frame)
        params = {key: args_info.locals[key] for key in args_info.args if key not in {"cache_dir", "filestore_dir"}}
        cache_key = create_consistent_hash(**params)
        cache_path = (
            cache_dir / "fl_resampled" / park_id / device_id / f"{start_dt.strftime('%Y%m%d')}_{cache_key}.parquet"
        )
        if cache_path.exists():
            logger.info("Reading: %s", cache_path)
            return pd.read_parquet(cache_path)
    result_df = make_fl_resampled_one_device(
        park_id=park_id,
        device_id=device_id,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        timebase_s=timebase_s,
        filestore_dir=filestore_dir,
        tags=tags,
        busy_tags=busy_tags,
        min_data_count=min_data_count,
        minmax_tags=minmax_tags,
    )
    if cache_dir is not None and not result_df.empty:
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        try:
            logger.info("Writing: %s", cache_path)
            result_df.to_parquet(cache_path)
        except ArrowInvalid as e:
            msg = f"Error saving resampled data to cache at {cache_path}: {e}"
            logger.exception(msg)
            csv_path = cache_path.with_stem(f"{cache_path.stem}_error").with_suffix(".csv")
            logger.info("Writing: %s", csv_path)
            result_df.to_csv(csv_path)
            msg = f"Saved resampled data to CSV at {csv_path} for troubleshooting."
            logger.info(msg)
            msg = f"Returning empty dataframe for {park_id=} {device_id=} {start_dt=}"
            logger.warning(msg)
            return pd.DataFrame()
    return result_df


def _generate_dates_in_range(start_dt: dt.datetime, end_dt_excl: dt.datetime) -> list[dt.date]:
    """Generate dates in a datetime range."""
    date_range = pd.date_range(
        start=start_dt.date(), end=(end_dt_excl - dt.timedelta(microseconds=1)).date(), freq="D", inclusive="both"
    )
    return [date.date() for date in date_range]


def get_fl_resampled_one_device(  # noqa: PLR0913
    *,
    park_id: str,
    device_id: str,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    timebase_s: int = 1,
    filestore_dir: Path | None = None,
    tags: Sequence[str] | None = None,
    busy_tags: Sequence[str] | None = None,
    minmax_tags: Sequence[str] | None = None,
    min_data_count: float | None = None,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Return resampled fastlog data for a single device over the given date range, chunked by day."""
    # chunk by day
    day_dfs = []
    for day in _generate_dates_in_range(start_dt, end_dt_excl):
        day_start_dt = pd.Timestamp(day)
        day_end_dt_excl = pd.Timestamp(day) + pd.DateOffset(days=1)
        day_df = _get_fl_resampled_one_device_one_day(
            park_id=park_id,
            device_id=device_id,
            start_dt=day_start_dt,
            end_dt_excl=day_end_dt_excl,
            timebase_s=timebase_s,
            filestore_dir=filestore_dir,
            tags=tags,
            busy_tags=busy_tags,
            minmax_tags=minmax_tags,
            min_data_count=min_data_count,
            cache_dir=cache_dir,
        )
        if not day_df.empty:
            day_dfs.append(day_df)
    if len(day_dfs) > 0:
        result_df = pd.concat(day_dfs)
        # convert result_df index to DateTimeIndex if necessary
        if result_df.index.tzinfo is None:  # type: ignore[attr-defined]
            # HOT FastLog is in UTC
            result_df.index = pd.to_datetime(result_df.index, utc=True)
        return (
            result_df[(result_df.index >= pd.Timestamp(start_dt)) & (result_df.index < pd.Timestamp(end_dt_excl))]
            .resample(f"{timebase_s}s")
            .last()
        )
    msg = f"No data found for {park_id=} {device_id=} between {start_dt=} and {end_dt_excl=}"
    logger.warning(msg)
    return pd.DataFrame()


def _get_tag_list_from_park_id(park_id: str) -> list[str]:
    if park_id in SIEMENS_PARKS:
        return SIEMENS_TAGS
    msg = f"{park_id=} not implemented"
    raise NotImplementedError(msg)


def _load_siemens_fastlog_files(  # noqa: C901, PLR0912
    *, park_id: str, filestore_dir: Path, wtgid: str, day_str: str, tags_to_load: list[str]
) -> pd.DataFrame:
    if not filestore_dir.is_dir():
        msg = f"{filestore_dir} is not a valid path"
        raise FileNotFoundError(msg)
    fl_data_dir = filestore_dir / "FL" / park_id / wtgid / day_str
    logger.info("Reading fastlog files for %s %s from: %s", wtgid, day_str, fl_data_dir)
    tags_df = pd.DataFrame()
    for tag in tags_to_load:
        prefix = "" if tag.startswith("computed_") else "Wtc_TDI_"
        str_for_file_search = f"FL{wtgid}_{prefix}{tag}_{day_str.replace('-', '_')}"
        found_file = False
        for file in fl_data_dir.glob("*.prq"):
            if str_for_file_search in file.name and not file.name.startswith(".azDownload"):
                logger.debug("Reading: %s", file)
                tag_df = pd.read_parquet(file)
                found_file = True
                break
        if not found_file:
            for file in fl_data_dir.glob("*.h5"):
                if str_for_file_search in file.name and not file.name.startswith(".azDownload"):
                    logger.debug("Reading: %s", file)
                    tag_df = cast("pd.DataFrame", pd.read_hdf(file))
                    found_file = True
                    break
        if not found_file:
            str_for_file_search = f"{wtgid}_{tag}_{day_str.replace('-', '_')}"
            for file in fl_data_dir.glob("*.h5"):
                if str_for_file_search in file.name and not file.name.startswith(".azDownload"):
                    logger.debug("Reading: %s", file)
                    tag_df = cast("pd.DataFrame", pd.read_hdf(file))
                    found_file = True
                    break
        if found_file:
            if tag_df.empty:
                continue
            tag_df = tag_df[~tag_df.index.duplicated(keep="last")].sort_index()
            if not isinstance(tag_df.index, pd.DatetimeIndex):
                msg = f"tag_df.index is not a DatetimeIndex. {type(tag_df.index)=}"
                raise TypeError
            tag_df.index.name = TIMESTAMP_NAME
            if len(tag_df.columns) > 1:
                msg = f"Found multiple columns in {file} for tag {tag}, only expected one"
                raise RuntimeError(msg)
            tag_df = tag_df.rename(columns={f"{prefix}{tag}": tag})  # type:ignore[call-overload]
            if tag_df.columns[0] != tag:
                msg = f"Expected column name {tag}, but got {tag_df.columns[0]}"
                raise RuntimeError(msg)
            tags_df = tags_df.join(tag_df, how="outer", sort=True)
        else:
            msg = f"could not find {tag} in {fl_data_dir}"
            logger.warning(msg)
            continue
    return tags_df


def _remove_multiple_columns_from_tag_df(*, tag_df: pd.DataFrame, tag: str) -> pd.DataFrame:
    if len(tag_df.columns) > 1:
        msg = f"{tag} raw file includes multiple columns ({tag_df.columns}). All columns other than {tag} removed."
        logger.warning(msg)
        return tag_df[[tag]]
    return tag_df


def _check_for_timestamps_a_month_dt_range(
    *, tag_df: pd.DataFrame, start_dt: dt.datetime, end_dt_excl: dt.datetime, tag: str
) -> None:
    """Raise warning if timestamps that fall a month outside the expected datetime range are present."""
    if (
        (tag_df.index < (start_dt - pd.DateOffset(months=1)))
        | (tag_df.index >= (end_dt_excl + pd.DateOffset(months=1)))
    ).sum() > 0:
        msg = (
            f"{tag} raw files include data a month outside the valid range of {start_dt=} to {end_dt_excl=}."
            f"This is likely due to a bug in the raw data."
        )
        logger.warning(msg)


def _get_raw_df_dict(  # noqa: PLR0913
    *,
    park_id: str,
    device_id: str,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    filestore_dir: Path | None = None,
    tags: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Get dictionary of raw FL dataframes by tag."""
    filestore_dir = get_filestore_dir() if filestore_dir is None else filestore_dir
    dates_in_range = _generate_dates_in_range(start_dt, end_dt_excl)

    tags = _get_tag_list_from_park_id(park_id) if tags is None else tags

    tag_df_dict = {}
    for tag in tags:
        tag_df_list = []
        for date in dates_in_range:
            if park_id in SIEMENS_PARKS:
                tag_df = _load_siemens_fastlog_files(
                    park_id=park_id,
                    filestore_dir=filestore_dir,
                    wtgid=device_id,
                    day_str=str(date),
                    tags_to_load=[tag],
                )
            if not tag_df.empty:
                tag_df_list.append(tag_df)
        if len(tag_df_list) == 0:
            tag_df_dict[tag] = pd.DataFrame()
        else:
            tag_df_list = [_remove_multiple_columns_from_tag_df(tag_df=tag_df, tag=tag) for tag_df in tag_df_list]
            tag_df = pd.concat(tag_df_list)
            tag_df = tag_df[~tag_df.index.duplicated(keep="last")].sort_index()
            if not tag_df.empty:
                _check_for_timestamps_a_month_dt_range(
                    tag_df=tag_df, start_dt=start_dt, end_dt_excl=end_dt_excl, tag=tag
                )
                tag_df = tag_df.loc[
                    (tag_df.index >= pd.Timestamp(start_dt)) & (tag_df.index < pd.Timestamp(end_dt_excl)), :
                ]
            tag_df_dict[tag] = tag_df
    return tag_df_dict


def make_fl_resampled_one_device(  # noqa: PLR0913
    *,
    park_id: str,
    device_id: str,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    timebase_s: int = 1,
    filestore_dir: Path | None = None,
    tags: Sequence[str] | None = None,
    busy_tags: Sequence[str] | None = None,
    minmax_tags: Sequence[str] | None = None,
    min_data_count: float | None = None,
) -> pd.DataFrame:
    """Load raw fastlog data and resample it to the target timebase for a single device."""
    raw_df_dict = _get_raw_df_dict(
        park_id=park_id,
        device_id=device_id,
        start_dt=start_dt - pd.Timedelta(days=1),
        end_dt_excl=end_dt_excl + pd.Timedelta(hours=1),
        filestore_dir=filestore_dir,
        tags=tags,
    )
    if len(raw_df_dict) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    msg = f"Resampling data for {device_id=} {start_dt=}"
    logger.info(msg)

    resampled_df = resample_fastlog_tags(
        raw_df_dict=raw_df_dict,
        timebase_s=timebase_s,
        busy_tags=busy_tags,
        minmax_tags=minmax_tags,
        min_data_count=min_data_count,
    )

    return (
        resampled_df[(resampled_df.index >= pd.Timestamp(start_dt)) & (resampled_df.index < pd.Timestamp(end_dt_excl))]
        .resample(f"{timebase_s}s")
        .last()
    )


def upsample_and_ffill_stopping_at_nans(
    *,
    tag_df: pd.DataFrame,
    timebase_s: int,
    subsampling_timebase_ms: int,
    only_ffill_one_timebase: bool,
    busy_tag_nan_times: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Upsample a DataFrame and forward-fill, stopping the forward fill at NaNs.

    Parameters
    ----------
    tag_df : pd.DataFrame
        The DataFrame to upsample.
    timebase_s : int
        The target timebase for subsequent resampling.
    subsampling_timebase_ms : int
        This function will upsample the DataFrame using this timebase.
    only_ffill_one_timebase : bool
        If True only limited forward fill within the horizon of timebase_s is applied,
        otherwise forward fill until NaN or busy_tag NaN is applied.
    busy_tag_nan_times : pd.DatetimeIndex | None
        DatetimeIndex of times when the busy tag is NaN and forward filling must stop

    """
    if tag_df.empty:
        return tag_df
    # insert nans into df at the busy_tag_nan_times
    if busy_tag_nan_times is not None:
        # figure out busy_tag_nan_times which are not in the index of df
        busy_tag_nan_times_to_add = busy_tag_nan_times.difference(tag_df.index.tolist())
        # add nans to df at busy_tag_nan_times
        tag_df = tag_df.reindex(tag_df.index.union(busy_tag_nan_times_to_add))
    upsampling_factor = timebase_s * 1000 // (subsampling_timebase_ms)
    ffill_limit = None if not only_ffill_one_timebase else upsampling_factor - 1
    return tag_df.resample(pd.Timedelta(milliseconds=subsampling_timebase_ms)).ffill(limit=ffill_limit)


def resample_fastlog_tags(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    raw_df_dict: dict[str, pd.DataFrame],
    timebase_s: int,
    busy_tags: Sequence[str] | None = None,
    ffill_tags: Sequence[str] | None = None,
    circular_tags: Sequence[str] | None = None,
    minmax_tags: Sequence[str] | None = None,
    min_data_count: float | None = None,
    min_data_count_tag: str | None = None,
) -> pd.DataFrame:
    """Resample all tags to the target timebase."""
    if busy_tags is None:
        siemens_typical_busy_tags = {"ActPower_Value", "AcWindSp_AcWindSp", "GenRpm_Value"}
        busy_tags = tuple(x for x in raw_df_dict if x in siemens_typical_busy_tags)
    if ffill_tags is None:
        ffill_tags = tuple(x for x in raw_df_dict if x not in busy_tags)
    if circular_tags is None:
        siemens_typical_circular_tags = {"YawPos_Value", "AcWindDr_Value"}
        res_typical_circular_tags = {
            "computed_driver_pre_processed_yaw_direction_true_degrees",
            "computed_core_post_processed_direction_for_wake_steering",
        }
        circular_tags = tuple(
            x for x in raw_df_dict if x in (siemens_typical_circular_tags | res_typical_circular_tags)
        )

    subsampling_timebase_ms = min(1000, timebase_s * 1000 // 20)
    busy_upsampled = pd.DataFrame(index=pd.DatetimeIndex([], name=TIMESTAMP_NAME))
    for tag, tag_df in raw_df_dict.items():
        if tag not in busy_tags or tag_df.empty:
            continue
        tag_upsampled = upsample_and_ffill_stopping_at_nans(
            tag_df=tag_df,
            timebase_s=timebase_s,
            subsampling_timebase_ms=subsampling_timebase_ms,
            only_ffill_one_timebase=True,
        )
        busy_upsampled = pd.merge_ordered(busy_upsampled, tag_upsampled, on=TIMESTAMP_NAME).set_index(TIMESTAMP_NAME)
    busy_resampled = busy_upsampled.resample(f"{timebase_s}s").mean()
    busy_tag_nan_times = busy_resampled.index[busy_resampled.isna().all(axis=1)]
    if not isinstance(busy_tag_nan_times, pd.DatetimeIndex):
        msg = f"Expected a DatetimeIndex, but got {type(busy_tag_nan_times)}"
        raise TypeError(msg)
    # disregard the first nan in a run of consecutive nans
    busy_tag_nan_times = busy_tag_nan_times[
        busy_tag_nan_times.diff().total_seconds() == timebase_s  # type:ignore[attr-defined]
    ]
    # upsample all tags using busy tag info to stop forward fill when busy tags are all nan
    upsampled = pd.DataFrame(index=pd.DatetimeIndex([], name=TIMESTAMP_NAME))
    for tag, tag_df in raw_df_dict.items():
        if tag_df.empty:
            continue
        tag_upsampled = upsample_and_ffill_stopping_at_nans(
            tag_df=tag_df,
            timebase_s=timebase_s,
            subsampling_timebase_ms=subsampling_timebase_ms,
            only_ffill_one_timebase=tag not in ffill_tags,
            busy_tag_nan_times=busy_tag_nan_times if len(busy_tag_nan_times) > 0 else None,
        )
        upsampled = pd.merge_ordered(upsampled, tag_upsampled, on=TIMESTAMP_NAME).set_index(TIMESTAMP_NAME)
    if upsampled.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=TIMESTAMP_NAME))
    circ_cols = [x for x in circular_tags if x in upsampled.columns]
    noncirc_cols = [x for x in upsampled.columns if x not in circ_cols]

    # Separate numeric and non-numeric columns
    noncirc_df = upsampled[noncirc_cols]
    numeric_cols = noncirc_df.select_dtypes(include="number").columns
    nonnumeric_cols = noncirc_df.select_dtypes(exclude="number").columns
    noncirc_resampled = pd.concat(
        [
            noncirc_df[numeric_cols].resample(f"{timebase_s}s").mean(),
            noncirc_df[nonnumeric_cols].resample(f"{timebase_s}s").last(),
        ],
        axis=1,
    )
    circ_resampled = circ_mean_resample_degrees(upsampled[circ_cols], resample_timedelta=pd.Timedelta(f"{timebase_s}s"))
    resampled_df = pd.merge_ordered(noncirc_resampled, circ_resampled, on=TIMESTAMP_NAME).set_index(TIMESTAMP_NAME)
    if minmax_tags is not None:
        minmax_tags_in_upsampled = [x for x in minmax_tags if x in upsampled.columns]
        if len(minmax_tags_in_upsampled) > 0:
            max_df = upsampled[minmax_tags_in_upsampled].resample(f"{timebase_s}s").max()
            max_df = max_df.rename(columns={x: f"max_{x}" for x in minmax_tags_in_upsampled})
            resampled_df = pd.merge_ordered(resampled_df, max_df, on=TIMESTAMP_NAME).set_index(TIMESTAMP_NAME)
            min_df = upsampled[minmax_tags_in_upsampled].resample(f"{timebase_s}s").min()
            min_df = min_df.rename(columns={x: f"min_{x}" for x in minmax_tags_in_upsampled})
            resampled_df = pd.merge_ordered(resampled_df, min_df, on=TIMESTAMP_NAME).set_index(TIMESTAMP_NAME)
    resampled_df.index = pd.DatetimeIndex(resampled_df.index, freq=f"{timebase_s}s")

    if min_data_count is not None:
        if min_data_count_tag is not None:
            tag_df = raw_df_dict[min_data_count_tag]
            tag_upsampled = upsample_and_ffill_stopping_at_nans(
                tag_df=tag_df,
                timebase_s=timebase_s,
                subsampling_timebase_ms=subsampling_timebase_ms,
                only_ffill_one_timebase=min_data_count_tag not in ffill_tags,
                busy_tag_nan_times=busy_tag_nan_times if len(busy_tag_nan_times) > 0 else None,
            )
            count_df = tag_upsampled.resample(f"{timebase_s}s").count()
        else:
            count_df = busy_upsampled.resample(f"{timebase_s}s").count()
        low_count_times = count_df.index[count_df.lt(min_data_count).all(axis=1)]
        resampled_df.loc[low_count_times, numeric_cols] = np.nan
        resampled_df.loc[low_count_times, circ_cols] = np.nan
        resampled_df.loc[low_count_times, nonnumeric_cols] = pd.NA
    return resampled_df


def create_consistent_hash(**kwargs) -> str:  # noqa: ANN003
    """Create a consistent hash from arguments, useful for caching.

    Uses SHA-256 but encodes the result in base64 instead of hexadecimal.
    This produces a 44-character string, which we then truncate to 32 characters for brevity.
    """
    all_args = [kwargs]

    def serialize(obj):  # noqa: ANN001 ANN202
        if isinstance(obj, int | float | str | bool | type(None)):
            return obj
        if isinstance(obj, list | tuple):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {str(key): serialize(value) for key, value in obj.items()}
        return str(obj)

    serialized = json.dumps(serialize(all_args), sort_keys=True)
    hash_bytes = hashlib.sha256(serialized.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(hash_bytes).decode("utf-8")[:32]
