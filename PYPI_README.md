# hot-open

Helpers for loading the [Hill of Towie open wind farm dataset](https://zenodo.org/records/22662930)
into pandas: 10-minute SCADA, ~1 Hz turbine fastlog, ZX300/ZXTM LiDAR, turbine metadata and ERA5
reanalysis. Input files are downloaded from Zenodo on first use, and intermediate results are cached
as parquet.

The package is developed alongside the analyses in
[resgroup/hill-of-towie-open-source-analysis](https://github.com/resgroup/hill-of-towie-open-source-analysis),
which uses it with [wind-up](https://github.com/resgroup/wind-up) to assess upgrades at Hill of Towie.

## Installation

```commandline
pip install hot-open
```

Python 3.10 or newer is required. The import name is `hot_open`.

## Example

```python
import pandas as pd

from hot_open.fastlog_helpers import load_hot_fl_data
from hot_open.scada_helpers import load_hot_10min_data
from hot_open.settings import get_data_dir, get_filestore_dir

start_dt = pd.Timestamp("2026-02-20", tz="UTC")
end_dt_excl = pd.Timestamp("2026-02-21", tz="UTC")

# 10-minute SCADA, wide format with (turbine, field) columns
scada_df = load_hot_10min_data(
    data_dir=get_data_dir(), wtg_numbers=[3, 7], start_dt=start_dt, end_dt_excl=end_dt_excl
)

# ~1 Hz controller fastlog for the same turbines
fastlog_df = load_hot_fl_data(
    data_dir=get_filestore_dir(), wtg_numbers=[3, 7], start_dt=start_dt, end_dt_excl=end_dt_excl
)
```

The first call downloads the files it needs from Zenodo (~12 GB for the fastlog).

## What's in the package

| Module | Main entry points |
| --- | --- |
| `hot_open.scada_helpers` | `load_hot_10min_data()`: 10-minute SCADA for selected turbines and dates |
| `hot_open.fastlog_helpers` | `load_hot_fl_data()`: ~1 Hz turbine fastlog, resampled and cached by day ([details](https://github.com/resgroup/hill-of-towie-open-source-analysis/blob/main/docs/fastlog_resampling.md)) |
| `hot_open.lidar_helpers` | `load_zx_lidar_10min_data()`, `load_zx_lidar_fl_data()`, `add_shear_and_veer()` |
| `hot_open.unpack` | `unpack_local_scada_data_v2()`, `unpack_local_meta_data()`: the whole SCADA dataset and metadata in [wind-up](https://github.com/resgroup/wind-up) format |
| `hot_open.era5_helpers` | `get_hot_era5_hourly_df()`, `get_hot_reanalysis_datasets()`: ERA5 for the site via the Open-Meteo archive API |
| `hot_open.sourcing_data` | `ensure_hot_data_files()`: download specific dataset files from Zenodo |

Field definitions, reference frames and known data quality issues are documented in
`Hill_of_Towie_data_dictionary.md` in the [Zenodo data pack](https://zenodo.org/records/22662930).

## Configuration

All paths are set by optional environment variables, which can also go in a `.env` file. If a
variable isn't set, a folder under `~/temp/hill-of-towie-open-source-analysis/` is used.

| Variable | Used for |
| --- | --- |
| `HOT_OPEN_DATA_DIR` | Downloaded dataset files |
| `HOT_OPEN_CACHE_DIR` | Parquet caches; safe to delete to force a rebuild |
| `HOT_OPEN_OUTPUT_DIR` | Per-script outputs |
| `HOT_OPEN_FILESTORE_DIR` | An existing fastlog `Filestore` tree; when set, the fastlog zip isn't downloaded |
| `WINDUP_OUTPUT_DIR` | wind-up assessment outputs |

## Versioning

Releases follow [semantic versioning](https://semver.org/). While the version is `0.x`, a minor
version bump may contain breaking changes. Release notes are on the
[GitHub releases page](https://github.com/resgroup/hill-of-towie-open-source-analysis/releases).

## License

BSD 3-Clause. Contact: Alex.Clerc@res-group.com
