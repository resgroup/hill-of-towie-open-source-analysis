"""Helpers for loading and resampling Siemens fastlog (FL) data from the Filestore directory tree."""

import base64
import datetime as dt
import hashlib
import inspect
import json
import logging
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid

from hot_open.circular_math import circ_mean_resample_degrees, circ_std_resample_degrees
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
    siemens_parks: set[str] | None = None,
    refresh_cache: bool = False,
    **resample_kwargs: object,
) -> pd.DataFrame:
    """Return resampled fastlog data for multiple devices as a multi-level (device_id, tag) DataFrame.

    Extra keyword arguments are forwarded to :func:`resample_fastlog_tags` and join the per-day
    cache key when they differ from its defaults.

    ``siemens_parks`` overrides the set of park ids loaded with the Siemens fastlog reader;
    when ``None`` the module default ``SIEMENS_PARKS`` is used. Pass it to load parks that are
    not part of the public dataset.

    The per-day cache self-invalidates when a day's source files are newer than its cached
    parquet (e.g. late/backfilled data), so partial recent days recover on a later run. Pass
    ``refresh_cache=True`` to force every day to be recomputed and overwritten regardless.
    """
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
            siemens_parks=siemens_parks,
            refresh_cache=refresh_cache,
            **resample_kwargs,
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


RawLoader = Callable[[pd.Timestamp, pd.Timestamp], dict[str, pd.DataFrame]]
"""Load raw per-tag frames covering ``[start, end)``, keyed by tag, nulls dropped.

The one thing the chunk/cache orchestration needs from a data source. Anything satisfying it
-- a filestore tree, a parquet file, a database -- can reuse that orchestration.
"""

# A chunk is loaded with context either side before being trimmed back, so a value carried in
# from before the chunk is available and nothing is bridged across its edges. The lead-in must
# exceed one output window, not merely a gap cutoff: resample_fastlog_tags disregards the first
# NaN in a run of busy-tag NaNs, so without a spare window the first window of every chunk that
# falls inside a long outage would escape being masked.
_CHUNK_LEAD_IN = pd.Timedelta(days=1)
_CHUNK_TRAIL = pd.Timedelta(hours=1)


def _resample_one_chunk(
    *,
    load_raw: RawLoader,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    timebase_s: int,
    **resample_kwargs: object,
) -> pd.DataFrame:
    """Load one chunk with context either side, resample it, then trim back to the chunk."""
    raw_df_dict = load_raw(pd.Timestamp(start_dt) - _CHUNK_LEAD_IN, pd.Timestamp(end_dt_excl) + _CHUNK_TRAIL)
    if len(raw_df_dict) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([]))

    resampled_df = resample_fastlog_tags(
        raw_df_dict=raw_df_dict,
        timebase_s=timebase_s,
        **resample_kwargs,  # type: ignore[arg-type]
    )
    return (
        resampled_df[(resampled_df.index >= pd.Timestamp(start_dt)) & (resampled_df.index < pd.Timestamp(end_dt_excl))]
        .resample(f"{timebase_s}s")
        .last()
    )


def _get_resampled_one_chunk_cached(  # noqa: PLR0913
    *,
    load_raw: RawLoader,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    timebase_s: int,
    cache_key_extra: dict,
    cache_subdir: tuple[str, ...] = (),
    source_mtime_fn: Callable[[pd.Timestamp, pd.Timestamp], float | None] | None = None,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    **resample_kwargs: object,
) -> pd.DataFrame:
    """Resample one chunk, reading from and writing to a per-chunk parquet cache."""
    if cache_dir is not None:
        # An explicit dict, not argument introspection: a RawLoader is a callable, and
        # create_consistent_hash falls back to str() for one, which embeds its memory address
        # and so changes every process. That would be a silent permanent cache miss. Only the
        # caller knows what identifies its source, hence cache_key_extra.
        key_params = {
            **cache_key_extra,
            "start_dt": start_dt,
            "end_dt_excl": end_dt_excl,
            "timebase_s": timebase_s,
            **_non_default_resample_kwargs(resample_kwargs),
        }
        cache_key = create_consistent_hash(**key_params)
        cache_path = cache_dir.joinpath(*cache_subdir) / f"{start_dt.strftime('%Y%m%d')}_{cache_key}.parquet"
        if refresh_cache:
            # Delete up-front so an empty recompute can't leave a stale parquet that later
            # (non-refresh) runs would silently reuse, undermining the intent of refresh_cache.
            cache_path.unlink(missing_ok=True)
        elif cache_path.exists():
            source_mtime = (
                None if source_mtime_fn is None else source_mtime_fn(pd.Timestamp(start_dt), pd.Timestamp(end_dt_excl))
            )
            # A freshly written parquet's mtime is later than every source file read to build it,
            # so an unchanged chunk stays a hit; a backfilled one (newer source mtime) recomputes.
            # source_mtime_fn=None means never invalidate, which is right for an immutable source.
            if source_mtime is None or cache_path.stat().st_mtime >= source_mtime:
                logger.info("Reading: %s", cache_path)
                return pd.read_parquet(cache_path)
            logger.info("Cache stale (source newer than cache); recomputing: %s", cache_path)

    result_df = _resample_one_chunk(
        load_raw=load_raw,
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        timebase_s=timebase_s,
        **resample_kwargs,
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
            msg = f"Returning empty dataframe for {cache_key_extra=} {start_dt=}"
            logger.warning(msg)
            return pd.DataFrame()
    return result_df


def get_resampled_chunked_cached(  # noqa: PLR0913
    *,
    load_raw: RawLoader,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    timebase_s: int = 1,
    cache_key_extra: dict,
    cache_subdir: tuple[str, ...] = (),
    source_mtime_fn: Callable[[pd.Timestamp, pd.Timestamp], float | None] | None = None,
    cache_dir: Path | None = None,
    refresh_cache: bool = False,
    **resample_kwargs: object,
) -> pd.DataFrame:
    """Resample a date range from any raw source, chunked by day with a per-day parquet cache.

    ``load_raw`` supplies the data, so this orchestration is not tied to any one source; see
    :data:`RawLoader`. ``get_fl_resampled_one_device`` is the Siemens filestore caller of it.

    ``cache_key_extra`` must carry whatever identifies the source, since only the caller knows
    it -- e.g. a park and device, or a file and turbine id. The date range and ``timebase_s`` are
    added here, and any ``resample_kwargs`` differing from ``resample_fastlog_tags``' defaults.

    ``source_mtime_fn`` returns the newest source mtime for a range, so a chunk whose source has
    been rewritten recomputes. Leave it ``None`` for an immutable source, which keeps cached
    chunks indefinitely.

    Extra keyword arguments are forwarded to :func:`resample_fastlog_tags`.
    """
    chunk_dfs = []
    for day in _generate_dates_in_range(start_dt, end_dt_excl):
        day_df = _get_resampled_one_chunk_cached(
            load_raw=load_raw,
            start_dt=pd.Timestamp(day),
            end_dt_excl=pd.Timestamp(day) + pd.DateOffset(days=1),
            timebase_s=timebase_s,
            cache_key_extra=cache_key_extra,
            cache_subdir=cache_subdir,
            source_mtime_fn=source_mtime_fn,
            cache_dir=cache_dir,
            refresh_cache=refresh_cache,
            **resample_kwargs,
        )
        if not day_df.empty:
            chunk_dfs.append(day_df)
    if len(chunk_dfs) == 0:
        return pd.DataFrame()
    result_df = pd.concat(chunk_dfs)
    range_tz = pd.Timestamp(start_dt).tz
    if result_df.index.tzinfo is None and range_tz is not None:  # type: ignore[attr-defined]
        # Chunks are cut on naive calendar dates, so they come back naive however the caller asked.
        # Localise rather than convert: a source's timestamps are in the clock its caller asks in
        # (HOT fastlog files carry naive UTC, and every HOT caller asks in UTC).
        result_df.index = result_df.index.tz_localize(range_tz)  # type: ignore[attr-defined]
    return (
        result_df[(result_df.index >= pd.Timestamp(start_dt)) & (result_df.index < pd.Timestamp(end_dt_excl))]
        .resample(f"{timebase_s}s")
        .last()
    )


def _siemens_raw_loader(
    *,
    park_id: str,
    device_id: str,
    filestore_dir: Path | None = None,
    tags: Sequence[str] | None = None,
    siemens_parks: set[str] | None = None,
) -> RawLoader:
    """Return a :data:`RawLoader` reading one device out of the Siemens fastlog filestore."""

    def load_raw(start_dt: pd.Timestamp, end_dt_excl: pd.Timestamp) -> dict[str, pd.DataFrame]:
        return _get_raw_df_dict(
            park_id=park_id,
            device_id=device_id,
            start_dt=start_dt,
            end_dt_excl=end_dt_excl,
            filestore_dir=filestore_dir,
            tags=tags,
            siemens_parks=siemens_parks,
        )

    return load_raw


def _siemens_source_mtime_fn(
    *, park_id: str, device_id: str, filestore_dir: Path | None = None
) -> Callable[[pd.Timestamp, pd.Timestamp], float | None]:
    """Return the source-freshness callback for one device's Siemens fastlog files."""

    def source_mtime(start_dt: pd.Timestamp, end_dt_excl: pd.Timestamp) -> float | None:
        return _max_source_mtime(
            park_id=park_id,
            device_id=device_id,
            start_dt=start_dt,
            end_dt_excl=end_dt_excl,
            filestore_dir=filestore_dir,
        )

    return source_mtime


def _generate_dates_in_range(start_dt: dt.datetime, end_dt_excl: dt.datetime) -> list[dt.date]:
    """Generate dates in a datetime range."""
    date_range = pd.date_range(
        start=start_dt.date(), end=(end_dt_excl - dt.timedelta(microseconds=1)).date(), freq="D", inclusive="both"
    )
    return [date.date() for date in date_range]


def _max_source_mtime(
    *,
    park_id: str,
    device_id: str,
    start_dt: dt.datetime,
    end_dt_excl: dt.datetime,
    filestore_dir: Path | None = None,
) -> float | None:
    """Return the newest mtime among the raw FL files feeding this day's resample, or None.

    Mirrors ``make_fl_resampled_one_device``'s read window (``start_dt - 1 day`` to
    ``end_dt_excl + 1 hour``) so a backfill into the neighbouring day folders also invalidates
    the cache. ``None`` means no source files were found (treated as "cannot be stale").
    """
    filestore_dir = get_filestore_dir() if filestore_dir is None else filestore_dir
    raw_start = start_dt - pd.Timedelta(days=1)
    raw_end_excl = end_dt_excl + pd.Timedelta(hours=1)
    mtimes: list[float] = []
    for date in _generate_dates_in_range(raw_start, raw_end_excl):
        day_dir = filestore_dir / "FL" / park_id / device_id / str(date)
        if not day_dir.is_dir():
            continue
        for file in day_dir.iterdir():
            if file.name.startswith(".azDownload") or file.suffix not in {".prq", ".h5"}:
                continue
            mtimes.append(file.stat().st_mtime)
    return max(mtimes) if mtimes else None


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
    siemens_parks: set[str] | None = None,
    refresh_cache: bool = False,
    **resample_kwargs: object,
) -> pd.DataFrame:
    """Return resampled fastlog data for a single device over the given date range, chunked by day.

    The Siemens filestore caller of :func:`get_resampled_chunked_cached`; the chunking, per-day
    parquet cache and source-freshness check all live there.

    Extra keyword arguments are forwarded to :func:`resample_fastlog_tags` and join the per-day
    cache key when they differ from its defaults.
    """
    result_df = get_resampled_chunked_cached(
        load_raw=_siemens_raw_loader(
            park_id=park_id,
            device_id=device_id,
            filestore_dir=filestore_dir,
            tags=tags,
            siemens_parks=siemens_parks,
        ),
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        timebase_s=timebase_s,
        # Reproduces, exactly, the parameter set the per-day key used to be built from by argument
        # introspection. Every entry is hashed whatever its value, including None, because that is
        # what the introspected version did: adding or dropping one here silently orphans every day
        # every HOT user has cached. TestSiemensWrapperOverGenericLayer pins the resulting key.
        cache_key_extra={
            "park_id": park_id,
            "device_id": device_id,
            "tags": tags,
            "busy_tags": busy_tags,
            "minmax_tags": minmax_tags,
            "min_data_count": min_data_count,
        },
        cache_subdir=("fl_resampled", park_id, device_id),
        source_mtime_fn=_siemens_source_mtime_fn(park_id=park_id, device_id=device_id, filestore_dir=filestore_dir),
        cache_dir=cache_dir,
        refresh_cache=refresh_cache,
        busy_tags=busy_tags,
        minmax_tags=minmax_tags,
        min_data_count=min_data_count,
        **resample_kwargs,
    )
    if result_df.empty:
        msg = f"No data found for {park_id=} {device_id=} between {start_dt=} and {end_dt_excl=}"
        logger.warning(msg)
        return pd.DataFrame()
    return result_df


def _get_tag_list_from_park_id(park_id: str, siemens_parks: set[str] | None = None) -> list[str]:
    parks = SIEMENS_PARKS if siemens_parks is None else siemens_parks
    if park_id in parks:
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
        prefix = "" if tag.startswith(("computed_", "alarms_")) else "Wtc_TDI_"
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
    siemens_parks: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Get dictionary of raw FL dataframes by tag."""
    filestore_dir = get_filestore_dir() if filestore_dir is None else filestore_dir
    dates_in_range = _generate_dates_in_range(start_dt, end_dt_excl)
    parks = SIEMENS_PARKS if siemens_parks is None else siemens_parks

    if park_id not in parks:
        msg = f"{park_id=} not in siemens_parks={parks}; pass siemens_parks= to load this park"
        raise NotImplementedError(msg)

    tags = _get_tag_list_from_park_id(park_id, siemens_parks=parks) if tags is None else tags

    tag_df_dict = {}
    for tag in tags:
        tag_df_list = []
        for date in dates_in_range:
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
    siemens_parks: set[str] | None = None,
    **resample_kwargs: object,
) -> pd.DataFrame:
    """Load raw fastlog data and resample it to the target timebase for a single device.

    The uncached single-range Siemens entry point; :func:`get_fl_resampled_one_device` is the
    chunked, cached one. Both read with the same lead-in, since both go through
    :func:`_resample_one_chunk`.

    Extra keyword arguments are forwarded to :func:`resample_fastlog_tags`, so options such as
    ``circular_tags``, ``ffill_tags``, ``std_tags``, ``min_raw_data_count``,
    ``busy_tag_ffill_limit_s`` and ``require_all_busy_tags`` are reachable from here. They are
    forwarded rather than enumerated deliberately: an enumerated list silently went stale as
    ``resample_fastlog_tags`` gained options, leaving several of them unreachable through the
    cache layer, and ``circular_tags`` in particular then fell back to a hardcoded tag-name set
    so a park named otherwise had its directions averaged across the 0/360 wrap.
    """
    msg = f"Resampling data for {device_id=} {start_dt=}"
    logger.info(msg)
    return _resample_one_chunk(
        load_raw=_siemens_raw_loader(
            park_id=park_id,
            device_id=device_id,
            filestore_dir=filestore_dir,
            tags=tags,
            siemens_parks=siemens_parks,
        ),
        start_dt=start_dt,
        end_dt_excl=end_dt_excl,
        timebase_s=timebase_s,
        busy_tags=busy_tags,
        minmax_tags=minmax_tags,
        min_data_count=min_data_count,
        **resample_kwargs,
    )


def _ffill_limit_from_seconds(*, ffill_limit_s: float, subsampling_timebase_ms: int, arg_name: str) -> int:
    """Convert a forward-fill horizon in seconds to a pandas ffill limit in sub-grid steps.

    Matches the one-timebase convention, so passing one timebase is a no-op.
    """
    limit = round(ffill_limit_s * 1000 / subsampling_timebase_ms) - 1
    if limit < 1:
        msg = (
            f"{arg_name}={ffill_limit_s} is not longer than the subsampling timebase "
            f"({subsampling_timebase_ms}ms), so nothing would be forward-filled"
        )
        raise ValueError(msg)
    return limit


# Grids the "auto" rule is allowed to choose, coarsest last. Quantising keeps the choice stable
# against day-to-day wobble in a tag's logging rate -- 100.0ms and 103.4ms must not produce two
# different grids for two days of the same signal -- and keeps the value legible in a log line.
_ALLOWED_SUBSAMPLING_MS = (10, 20, 25, 50, 100, 125, 200, 250, 500, 1000)

# The coarsest grid the auto rule will pick, whatever the logging rate says. A grid slower than 1Hz
# stops resolving when a value changed, so duration weighting degrades into "whichever sample
# happened to be latest"; 1Hz is also a common rate for a controller's own periodic statistics, so
# a product built on it stays comparable with them.
_MAX_AUTO_SUBSAMPLING_MS = 1000


def _median_logging_interval_ms(*, raw_df_dict: dict[str, pd.DataFrame], tags: Sequence[str]) -> float | None:
    """Return the fastest median inter-sample interval among ``tags``, in ms, or None if unknown.

    The fastest rather than the average: a grid coarser than the quickest tag throws that tag's
    samples away before anything is aggregated, which is what the rate-aware grid exists to stop.
    """
    intervals = []
    for tag in tags:
        tag_df = raw_df_dict.get(tag)
        if tag_df is None or len(tag_df) < 2:  # noqa: PLR2004
            continue
        median_s = pd.Series(tag_df.index).diff().dt.total_seconds().median()
        if pd.notna(median_s) and median_s > 0:
            intervals.append(median_s * 1000)
    return min(intervals) if intervals else None


def _resolve_subsampling_timebase_ms(
    *,
    subsampling_timebase_ms: int | str | None,
    timebase_s: int,
    raw_df_dict: dict[str, pd.DataFrame],
    busy_tags: Sequence[str],
) -> int:
    """Decide the sub-sampling grid: the legacy formula, an explicit value, or the measured rate.

    ``None`` keeps ``min(1000, timebase_s * 1000 // 20)``, which is keyed on the output timebase
    alone and so drifts away from the data in both directions: at ``timebase_s=600`` it samples a
    100ms tag at 1Hz and discards nine samples in ten, and at ``timebase_s=1`` it builds a 50ms
    grid for a signal logged every 12s, which is 240x more cells than there is information in them.

    ``"auto"`` measures the busy tags instead and takes the coarsest allowed grid no slower than
    the quickest of them, capped at 1s and at the output window itself. Only the busy tags are
    measured, per the design: a non-busy tag logging faster than every busy tag would be sampled
    down, which is the price of not scanning every tag on every chunk.
    """
    legacy = min(1000, timebase_s * 1000 // 20)
    if subsampling_timebase_ms is None:
        return legacy
    if subsampling_timebase_ms == "auto":
        fastest_ms = _median_logging_interval_ms(raw_df_dict=raw_df_dict, tags=busy_tags)
        if fastest_ms is None:
            # Nothing to measure (no busy tag, or a single sample): the timebase is all we know.
            return legacy
        ceiling = min(_MAX_AUTO_SUBSAMPLING_MS, timebase_s * 1000)
        allowed = [x for x in _ALLOWED_SUBSAMPLING_MS if x <= min(fastest_ms, ceiling)]
        chosen = max(allowed) if allowed else min(_ALLOWED_SUBSAMPLING_MS)
        msg = f"sub-sampling grid {chosen}ms from a fastest busy-tag interval of {fastest_ms:.0f}ms"
        logger.info(msg)
        return chosen
    if not isinstance(subsampling_timebase_ms, int) or isinstance(subsampling_timebase_ms, bool):
        msg = f"subsampling_timebase_ms must be an int, 'auto' or None, got {subsampling_timebase_ms!r}"
        raise TypeError(msg)
    if not 1 <= subsampling_timebase_ms <= timebase_s * 1000:
        msg = (
            f"subsampling_timebase_ms={subsampling_timebase_ms} must be between 1 and the output "
            f"window itself ({timebase_s * 1000}ms)"
        )
        raise ValueError(msg)
    return subsampling_timebase_ms


def _full_resample_grid(*, raw_df_dict: dict[str, pd.DataFrame], timebase_s: int) -> pd.DatetimeIndex:
    """Resample grid spanning every tag, for masks that must cover tags the busy frame lacks."""
    spans = [x.index for x in raw_df_dict.values() if not x.empty]
    if not spans:
        return pd.DatetimeIndex([], name=TIMESTAMP_NAME)
    freq = f"{timebase_s}s"
    start = min(x[0] for x in spans).floor(freq)
    end = max(x[-1] for x in spans).floor(freq)
    return pd.date_range(start, end, freq=freq, name=TIMESTAMP_NAME)


def upsample_and_ffill_stopping_at_nans(  # noqa: PLR0913
    *,
    tag_df: pd.DataFrame,
    timebase_s: int,
    subsampling_timebase_ms: int,
    only_ffill_one_timebase: bool,
    busy_tag_nan_times: pd.DatetimeIndex | None = None,
    ffill_limit_s: float | None = None,
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
    ffill_limit_s : float | None
        Forward-fill horizon in seconds, overriding only_ffill_one_timebase. For tags whose
        reporting interval is unrelated to timebase_s, e.g. 10s OPC logs on a 1s grid.
        ffill_limit_s=timebase_s reproduces only_ffill_one_timebase=True.

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
    ffill_limit: int | None = upsampling_factor - 1 if only_ffill_one_timebase else None
    if ffill_limit_s is not None:
        ffill_limit = _ffill_limit_from_seconds(
            ffill_limit_s=ffill_limit_s, subsampling_timebase_ms=subsampling_timebase_ms, arg_name="ffill_limit_s"
        )
    freq = pd.Timedelta(milliseconds=subsampling_timebase_ms)
    upsampled = tag_df.resample(freq).ffill(limit=ffill_limit)
    last_ts = tag_df.index[-1]
    if last_ts > upsampled.index[-1]:
        upsampled = pd.concat([upsampled, tag_df.iloc[[-1]]])
    return upsampled


def resample_fastlog_tags(  # noqa: C901, PLR0912, PLR0913, PLR0915
    *,
    raw_df_dict: dict[str, pd.DataFrame],
    timebase_s: int,
    busy_tags: Sequence[str] | None = None,
    ffill_tags: Sequence[str] | None = None,
    circular_tags: Sequence[str] | None = None,
    minmax_tags: Sequence[str] | None = None,
    std_tags: Sequence[str] | None = None,
    min_data_count: float | None = None,
    min_data_count_tag: str | None = None,
    min_raw_data_count: float | None = None,
    busy_tag_ffill_limit_s: float | None = None,
    require_all_busy_tags: bool = False,
    subsampling_timebase_ms: int | str | None = None,
) -> pd.DataFrame:
    """Resample all tags to the target timebase.

    ``busy_tag_ffill_limit_s`` is how long a busy tag's value stays representative, in seconds.
    It defaults to one timebase, which suits fastlog but not slower logging such as OPC: 10s data
    on a 1s grid leaves a busy tag absent from 9 cells in 10, so every window reads as an outage.

    ``require_all_busy_tags`` makes a window an outage when *any* busy tag is missing rather than
    only when all are. Use it where busy tags fail independently -- a vane channel dying while
    power keeps logging, which the default would forward-fill every other tag across.

    ``std_tags`` emits a ``std_<tag>`` column per named tag, computed on the same upsampled grid
    as the mean so it is duration weighted and consistent with the other aggregates. Tags also in
    ``circular_tags`` get the circular standard deviation, since a plain one is meaningless across
    the 0/360 wrap.

    ``subsampling_timebase_ms`` is the fine grid everything is upsampled onto before aggregation.
    ``None`` keeps the historical ``min(1000, timebase_s * 1000 // 20)``, which knows the output
    timebase but not the logging rate. ``"auto"`` measures the busy tags and follows them, capped at
    1s; an explicit value overrides both. See :func:`_resolve_subsampling_timebase_ms`.

    ``min_data_count`` counts non-NaN *sub-grid cells* of the busy tags, so it saturates once
    ``busy_tag_ffill_limit_s`` is set: with a 45s horizon a 60s window starved from 5 raw samples
    to 2 still reports 60 filled cells, indistinguishable from a healthy one. Use
    ``min_raw_data_count`` to require a minimum number of *raw* samples per window instead, which
    is what catches a run of short gaps that never individually trip the outage check. Its polarity
    follows ``require_all_busy_tags``: with that set, any busy tag below the threshold masks the
    window, otherwise all of them must be.
    """
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

    resolved_subsampling_ms = _resolve_subsampling_timebase_ms(
        subsampling_timebase_ms=subsampling_timebase_ms,
        timebase_s=timebase_s,
        raw_df_dict=raw_df_dict,
        busy_tags=busy_tags,
    )
    if busy_tag_ffill_limit_s is not None:
        _ffill_limit_from_seconds(
            ffill_limit_s=busy_tag_ffill_limit_s,
            subsampling_timebase_ms=resolved_subsampling_ms,
            arg_name="busy_tag_ffill_limit_s",
        )
    busy_upsampled = pd.DataFrame(index=pd.DatetimeIndex([], name=TIMESTAMP_NAME))
    for tag, tag_df in raw_df_dict.items():
        if tag not in busy_tags or tag_df.empty:
            continue
        tag_upsampled = upsample_and_ffill_stopping_at_nans(
            tag_df=tag_df,
            timebase_s=timebase_s,
            subsampling_timebase_ms=resolved_subsampling_ms,
            only_ffill_one_timebase=True,
            ffill_limit_s=busy_tag_ffill_limit_s,
        )
        busy_upsampled = pd.merge_ordered(busy_upsampled, tag_upsampled, on=TIMESTAMP_NAME).set_index(TIMESTAMP_NAME)
    busy_nan = busy_upsampled.resample(f"{timebase_s}s").mean().isna()
    if require_all_busy_tags:
        # A busy tag that is absent from raw_df_dict, or holds an empty frame, never became a
        # column above -- so without this it would be silently treated as not required, which is
        # the one case this flag exists for.
        absent = [x for x in busy_tags if x not in busy_nan.columns]
        if absent:
            busy_nan = busy_nan.reindex(_full_resample_grid(raw_df_dict=raw_df_dict, timebase_s=timebase_s))
            for tag in absent:
                busy_nan[tag] = True
        busy_tag_nan_times = busy_nan.index[busy_nan.any(axis=1)]
    else:
        busy_tag_nan_times = busy_nan.index[busy_nan.all(axis=1)]
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
            subsampling_timebase_ms=resolved_subsampling_ms,
            only_ffill_one_timebase=tag not in ffill_tags,
            busy_tag_nan_times=busy_tag_nan_times if len(busy_tag_nan_times) > 0 else None,
            ffill_limit_s=busy_tag_ffill_limit_s if tag in busy_tags else None,
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
    if std_tags is not None:
        std_tags_in_upsampled = [x for x in std_tags if x in upsampled.columns]
        # Circular tags need the resultant-length formula; a plain std is meaningless across 0/360.
        circ_std_tags = [x for x in std_tags_in_upsampled if x in circ_cols]
        linear_std_tags = [x for x in std_tags_in_upsampled if x not in circ_cols]
        std_frames = []
        if linear_std_tags:
            std_frames.append(upsampled[linear_std_tags].resample(f"{timebase_s}s").std())
        if circ_std_tags:
            std_frames.append(
                circ_std_resample_degrees(upsampled[circ_std_tags], resample_timedelta=pd.Timedelta(f"{timebase_s}s"))
            )
        for std_df in std_frames:
            std_df = std_df.rename(columns={x: f"std_{x}" for x in std_df.columns})  # noqa: PLW2901
            resampled_df = pd.merge_ordered(resampled_df, std_df, on=TIMESTAMP_NAME).set_index(TIMESTAMP_NAME)
    resampled_df.index = pd.DatetimeIndex(resampled_df.index, freq=f"{timebase_s}s")

    if min_data_count is not None:
        if min_data_count_tag is not None:
            tag_df = raw_df_dict[min_data_count_tag]
            tag_upsampled = upsample_and_ffill_stopping_at_nans(
                tag_df=tag_df,
                timebase_s=timebase_s,
                subsampling_timebase_ms=resolved_subsampling_ms,
                only_ffill_one_timebase=min_data_count_tag not in ffill_tags,
                busy_tag_nan_times=busy_tag_nan_times if len(busy_tag_nan_times) > 0 else None,
                ffill_limit_s=busy_tag_ffill_limit_s if min_data_count_tag in busy_tags else None,
            )
            count_df = tag_upsampled.resample(f"{timebase_s}s").count()
        else:
            count_df = busy_upsampled.resample(f"{timebase_s}s").count()
        low_count_times = count_df.index[count_df.lt(min_data_count).all(axis=1)]  # type:ignore[call-overload,arg-type]
        resampled_df.loc[low_count_times, numeric_cols] = np.nan
        resampled_df.loc[low_count_times, circ_cols] = np.nan
        resampled_df.loc[low_count_times, nonnumeric_cols] = pd.NA

    if min_raw_data_count is not None:
        # Raw samples per window, not sub-grid cells: see this function's docstring for why the
        # cell count cannot express this once busy_tag_ffill_limit_s is set.
        raw_counts = pd.DataFrame(index=resampled_df.index)
        for tag in busy_tags:
            raw_tag_df = raw_df_dict.get(tag)
            if raw_tag_df is None or raw_tag_df.empty:
                # An absent or empty busy tag has no samples, so it fails the threshold. Matches
                # require_all_busy_tags, which also treats such a tag as missing rather than absent.
                raw_counts[tag] = 0
            else:
                counts = raw_tag_df[tag].resample(f"{timebase_s}s").count()
                raw_counts[tag] = counts.reindex(resampled_df.index, fill_value=0)
        below = raw_counts.lt(min_raw_data_count)
        low_raw_times = raw_counts.index[below.any(axis=1) if require_all_busy_tags else below.all(axis=1)]
        # Blanks the derived std_/min_/max_ columns too, so a masked window keeps no aggregate.
        numeric_all = resampled_df.select_dtypes(include="number").columns
        nonnumeric_all = resampled_df.select_dtypes(exclude="number").columns
        resampled_df.loc[low_raw_times, numeric_all] = np.nan
        resampled_df.loc[low_raw_times, nonnumeric_all] = pd.NA
    return resampled_df


def _non_default_resample_kwargs(resample_kwargs: dict) -> dict:
    """Return only the resample options that differ from ``resample_fastlog_tags``' defaults.

    Cache keys must change when aggregation settings change, but omitting defaulted options is
    what keeps keys written before an option existed valid. Hashing every option instead would
    mean that merely widening ``resample_fastlog_tags``' signature orphaned every cached day,
    over parameters whose defaults reproduce the previous behaviour exactly.
    """
    defaults = {
        name: param.default
        for name, param in inspect.signature(resample_fastlog_tags).parameters.items()
        if param.default is not inspect.Parameter.empty
    }
    return {k: v for k, v in resample_kwargs.items() if k not in defaults or v != defaults[k]}


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
