# Resampling fastlog data

The fastlog in the Hill of Towie datapack is logged on change, at up to 30 Hz, one file per
turbine, tag and day. `hot_open.fastlog_helpers` turns it into a regular timebase (1 s, 60 s,
600 s, ...) with the same aggregates a controller's own periodic statistics carry. This page
describes how that resampling works and how to use it on data that is not in the Siemens filestore.

## Entry points

| Function | Use it for |
| --- | --- |
| `load_hot_fl_data()` | The Hill of Towie dataset at 1 s, by turbine number. Downloads from Zenodo on first use. |
| `get_fl_resampled()` | Any timebase, tag list and aggregation options for one or more Siemens devices, chunked by day and cached. |
| `get_resampled_chunked_cached()` | The same chunking and cache for **any raw source**: you supply a `load_raw` callable. |
| `resample_fastlog_tags()` | The aggregation itself, on raw frames already in memory. No chunking or cache. |

## How a window is aggregated

1. Every tag is upsampled onto a fine sub-grid (`subsampling_timebase_ms`) and forward filled, so
   a value logged on change counts for as long as it was current. Aggregates are therefore
   **duration weighted**, not sample weighted.
2. The forward fill of a *busy tag* (one that logs continuously, e.g. power or wind speed) is
   limited to `busy_tag_ffill_limit_s`, one timebase by default. A window where every busy tag
   is missing (any, with `require_all_busy_tags=True`) is an outage: nothing is filled across it
   and every tag reads NaN there.
3. Each window then takes the mean of every numeric tag, the circular mean of every tag in
   `circular_tags`, and the last value of non-numeric tags. `minmax_tags` add `min_<tag>` and
   `max_<tag>`; `std_tags` add `std_<tag>`, circular for direction tags.
4. `min_data_count` (filled sub-grid cells) or `min_raw_data_count` (raw samples per window)
   blank every column of a window with too little data behind it. Prefer the raw count once a
   fill limit is set, because filled cells saturate.

By default (`subsampling_timebase_ms="auto"`) the sub-grid follows the busy tags' logging rate,
measured per chunk: the coarsest grid that still resolves them, capped at 1 s and at half the
output window. With no busy tag to measure it falls back to `min(1000, timebase_s * 1000 // 20)`.
An explicit value must divide the output window; the grid is part of the cache key either way.

A fine grid costs time: on one turbine-day resampled to 600 s, the auto grid (25 ms there) took
about 16x as long as a 1 s grid. Pass `subsampling_timebase_ms=1000` for a coarse, fast grid when
only means matter. On the validation day below, a 1 s grid moves the median `ActPower_Value` mean
by under 0.05 kW, but it misses excursions shorter than a second, so extremes read inward: the
median minimum ~4 kW too high and the median maximum ~8 kW too low.

`source_clock_offset_s` corrects a source whose clock is known to be wrong: it is how many seconds
the source's clock reads ahead of true time, negative for one running behind. The raw index is
shifted before aggregating, so the output windows are labelled in true time.

## Validation against the turbine's own 10-minute statistics

`tests/test_fl_reproduces_10min.py` resamples one turbine-day of fastlog to 600 s and compares it
with the mean, min, max and standard deviation the controller computed at the time from the same
signal. Worst absolute difference over the day:

| 10-min field | fastlog tag | mean | min | max | stddev |
| --- | --- | ---: | ---: | ---: | ---: |
| `wtc_ActPower` (kW) | `ActPower_Value` | 0.09 | 1.0 | 2.0 | 0.09 |
| `wtc_AcWindSp` (m/s) | `AcWindSp_AcWindSp` | 0.003 | 0.0 | 0.0 | 0.002 |
| `wtc_NacelPos` (°) | `YawPos_Value` | 0.005 | 0.0 | 0.0 | 0.03 |
| `wtc_PitcPosA` (°) | `PitcPosA_Value` | 0.001 | 0.0 | 0.0 | 0.001 |

A second day with 23 periods crossing the 0/360° wrap agrees to 0.01° on the circular mean and
0.03° on the circular standard deviation, so both sides average angles circularly. Power is
quantised to 1 kW in the 10-minute record, which is why its extremes differ by whole numbers.

The fixture days are fetched from the Zenodo record by HTTP range request on first run (a few MB
out of the 12 GB fastlog archive) and cached in the data directory. Offline, the module is skipped.

## Using the chunked cache on your own source

```python
import pandas as pd
from hot_open.fastlog_helpers import get_resampled_chunked_cached

def load_raw(start: pd.Timestamp, end_excl: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Return {tag: frame} covering [start, end_excl), one column per frame, nulls dropped."""
    frame = pd.read_parquet("my_logger.parquet")  # naive DatetimeIndex, same clock as the bounds
    window = frame[(frame.index >= start) & (frame.index < end_excl)]
    return {tag: window[[tag]].dropna() for tag in window.columns}

df = get_resampled_chunked_cached(
    load_raw=load_raw,
    start_dt=pd.Timestamp("2026-03-01"),
    end_dt_excl=pd.Timestamp("2026-03-08"),
    timebase_s=60,
    cache_key_extra={"source": "my_logger", "turbine": "T01"},
    cache_subdir=("my_logger", "T01"),
    cache_dir=my_cache_dir,
    busy_tags=("power",),
    circular_tags=("wind_direction",),
    std_tags=("power", "wind_direction"),
)
```

Rules the chunking imposes:

- `start_dt`/`end_dt_excl` are naive, or tz-aware at a zero UTC offset throughout the range.
  Chunks are cut on calendar dates, so a range crossing a daylight-saving change is refused.
- `load_raw` receives naive bounds in the source's own clock and must return naive frames. The
  result is localised to the caller's timezone at the end.
- `timebase_s` must divide a day.
- `cache_key_extra` must identify the source: it is hashed into every per-day cache file name,
  along with the date, the timebase and every resample option that differs from its default.
- Pass `source_mtime_fn` to invalidate cached days when the source is rewritten (backfills). It
  is given the range actually read, which includes a day of lead-in and an hour of trail. Leave
  it `None` for an immutable source.

`siemens_raw_loader()` is the `load_raw` the Hill of Towie path uses; wrap it to derive a tag from
the raw signal while keeping the chunking, cache and clock correction.
