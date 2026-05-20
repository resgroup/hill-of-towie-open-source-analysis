# Hill of Towie - Open Source Dataset

Analysis of the Hill of Towie wind farm open dataset: https://zenodo.org/records/20204946

This repo contains analysis and helper functions to:
- download the [open dataset from Zenodo](https://zenodo.org/records/20204946) which includes 10-minute SCADA, ~1Hz fastlog collected by [Anemo](https://www.res-group.com/digital-solutions/anemo/), controller telemetry and LiDAR data.
- estimate energy uplift thanks to upgrades using the [wind-up](https://github.com/resgroup/wind-up) library
- ad-hoc analysis of the [Dynamic Yaw](https://www.res-group.com/digital-solutions/dynamic-yaw/) trial, including wake-steering event finding and LiDAR analysis.

The repository currently covers three upgrade validation campaigns:
- the [Dynamic Yaw](https://www.res-group.com/digital-solutions/dynamic-yaw/) wake steering and collective yaw control trial, analysis first published in 2026 (`scripts/wfc_analysis_2026`)
- the earlier [AeroUp](https://www.res-group.com/digital-solutions/aeroup/) and [TuneUp](https://www.res-group.com/digital-solutions/tuneup/) turbine upgrades, analysis first published in 2025 (`scripts/uplift_analysis_2025`)

## Quickstart

Analysis scripts in this repo download the input data from Zenodo automatically on first
use — no manual data staging is required. See `scripts/wfc_analysis_2026/inspect_data.py`
for a short example of loading the open dataset into pandas dataframes.

## Dynamic Yaw analysis

The folder `scripts/wfc_analysis_2026` analyses the energy uplift from the
[Dynamic Yaw](https://www.res-group.com/digital-solutions/dynamic-yaw/) **(DY)** controller trial at Hill of Towie. DY combines two control features which both provide energy uplift in different ways:

- **Wake Steering (WS)** — selected upwind turbines steer to
  deflect their wake away from a downwind partner. The controller applies a non-zero
  wake-steering offset only when the wind direction falls inside one of the pre-defined
  steering windows in `controller_config/wake-steering-lookup.csv`.
- **Collective Control (CC)** — every controlled turbine gets an improved wind direction signal based on pooled information from all turbines. This helps the turbine ignore turbulence when deciding whether to yaw or not, which can reduce yaw activity and increase energy yield. This is also essential for steering turbines to mitigate sensor bias during the steer.

DY is switched on and off by a synthetic toggle generated from UTC time (50-minute
on/off period, see `hot_dy_toggle_df()` in `uplift_ws.py`). The toggle pattern
lets wind-up estimate uplift from interleaved on/off pairs rather than a single
before/after split.

The driver script is `scripts/wfc_analysis_2026/total_uplift.py`. It runs the WS
and CC analyses end-to-end, generates many plots including all the plots shown below, and combines the two
results into a long-term AEP figure via `compute_lt_uplift()`.

### Site layout

The bubble plot
below shows the layout and ground elevation of each turbine — the west cluster sits
on the higher, exposed hilltop while the east cluster sits on a second smaller hill:

<img width="1440" height="922" alt="Hill of Towie turbine elevation" src="https://github.com/user-attachments/assets/21a8424a-3022-4c66-aa94-60d0933418ab" />

### Wake Steering result

`scripts/wfc_analysis_2026/uplift_ws.py` runs one wind-up analysis per
steering window. Each window uses a wind direction filter that matches the
upwind turbine's steering range. The P50 wake-steering uplift across the
qualifying steering events is **+1.32% (σ = 0.54%)**.

The plot below shows the per-turbine WS uplift across the west of the farm:

<img width="1440" height="1529" alt="Hill of Towie WS test turbine uplift" src="https://github.com/user-attachments/assets/a8e45437-53b2-478f-9d05-f11deb7e3363" />

### Collective Control result

`scripts/wfc_analysis_2026/uplift_cc.py` runs wind-up on the 10-min
periods that the per-steer analysis did *not* consume, isolating the contribution of
CC to uplift. T07 is excluded from the CC test set because it was not actively controlled during the campaign.

Across the 13 CC test turbines the P50 uplift is **+0.56% (σ = 0.47%)** with an
average yaw-activity reduction of **−6.0%**. The headline test-vs-reference bar
chart shows ~0% uplift on the four references (three reference turbines and the ZX300 LiDAR) as expected, while the test
group is positive:

<img width="1440" height="922" alt="Hill of Towie CC test turbine uplift" src="https://github.com/user-attachments/assets/dd28ddc9-f577-43ad-b7d3-a6c13dcb31f4" />

<img width="640" height="480" alt="combined uplift and 90pct CI" src="https://github.com/user-attachments/assets/2a43daad-0b95-45a3-b7d4-c1db871b96dd" />

CC also changes how much each turbine yaws. The bubble plot below shows the
per-turbine yaw activity change: blue means the turbine yawed less under CC:

<img width="1440" height="922" alt="Hill of Towie CC test turbine yaw activity change" src="https://github.com/user-attachments/assets/86a32515-5de6-4ffa-81ee-642d9760a787" />

### Combined long-term AEP

`compute_lt_uplift()` in `total_uplift.py` blends the WS and CC results into a
long-term AEP figure using FLORIS-derived fractions of time and energy that the
long-term steering schedule will spend in each mode. The combination gives:

- **AEP uplift P50: +0.7%**, σ = 0.4%, P95 = 0.0%
- Long-term yaw-activity change: **−2.4%** wind farm average, with an individual
  turbine maximum of **+8.1%** on T03 (the most actively-steered turbine).

### `scripts/wfc_analysis_2026` also includes:

- `inspect_data.py` — quick sanity plots of all data sources together.
- `zx300_wake_steers.py`, `zxtm_wake_steers.py` — wake steer plots with data from the
  on-site ZX300 (near T11) and ZXTM (on T07) LiDARs.
- `z300_wake_steer_animation.py`, `zxtm_wake_steer_animation.py` — matplotlib
  animations of LiDAR and turbine data during steering events.

## Earlier upgrade analysis (published in 2025)

The folder `scripts/uplift_analysis_2025` uses [wind-up](https://github.com/resgroup/wind-up) to run two analyses of energy uplift
after turbine upgrades.

Note any analysis can be run without running the others, e.g. you do not need to run `northing.py` before `aero_up.py`.

The script `scripts/uplift_analysis_2025/northing.py` calculates the northing corrections saved to
`scripts/uplift_analysis_2025/wind_up_config/northing/optimized_northing_corrections.yaml` and used in subsequent analyses. The plot
below shows the circular difference of ERA5 wind direction to each turbine's yaw direction before the northing
correction. The turbine north calibrations are apparently often wrong, which means the northing correction is quite
important.

![northing error vs reanalysis_wd before northing](https://github.com/user-attachments/assets/aaf1e4c6-dc10-4c59-9281-1051128464af)

### AeroUp

The script `scripts/uplift_analysis_2025/aero_up.py` analyses the energy uplift thanks
to [AeroUp](https://www.res-group.com/digital-solutions/aeroup/) for T13.
The result is a P50 uplift of 4.3% with a 90% confidence interval of 3.3% to 5.3%. This is visualized in the plot below
along with the uplift results for the three selected reference turbines, which are near 0% uplift as expected:

![combined uplift and 90% CI](https://github.com/user-attachments/assets/a36214d1-7308-4a16-9be3-9e2230170708)

### TuneUp

The script `scripts/uplift_analysis_2025/tune_up.py` analyses the energy uplift thanks
to [TuneUp](https://www.res-group.com/digital-solutions/tuneup/) for nine test turbines. The result is a P50 uplift of
1.1% with a 90% confidence interval of 0.2% to 2.0%. This is visualized in the plot below along with the uplift result for
the ten unchanged reference turbines, which is near 0% uplift as expected:

![combined uplift and 90% CI](https://github.com/user-attachments/assets/00b17f01-54ea-4532-ab1c-16921df0b70e)

## Python environment

The environment can be created and managed using [uv](https://docs.astral.sh/uv/). To create the environment:

```commandline
uv sync
```

To run formatting and linting:

```commandline
uv run poe all
```

## Configuration

All filesystem paths used by the scripts and helpers are controlled by environment
variables. Each variable is optional — if unset, the default location under your
home directory is used and created on first access. Values can either be exported
in your shell or written to a `.env` file at the repo root (auto-loaded via
`python-dotenv`).

| Variable | Default | Used for |
| --- | --- | --- |
| `HOT_OPEN_DATA_DIR` | `~/temp/hill-of-towie-open-source-analysis/data` | Input data: Zenodo downloads land here (SCADA zips, LiDAR data, fastlog data, metadata CSV). |
| `HOT_OPEN_OUTPUT_DIR` | `~/temp/hill-of-towie-open-source-analysis/output` | Per-script output directory (one subfolder per script, named after the script's filename). |
| `HOT_OPEN_CACHE_DIR` | `~/temp/hill-of-towie-open-source-analysis/cache` | Parquet caches for unpacked SCADA, fastlog-by-day, ERA5, and wind-up intermediates. Safe to delete to force a rebuild. |
| `HOT_OPEN_FILESTORE_DIR` | `<HOT_OPEN_DATA_DIR>/turbine_fastlog/Filestore` | Fastlog tree (`FL/<park>/<device>/<date>/`). When set explicitly, `load_hot_fl_data()` skips the Zenodo download of `turbine_fastlog.zip`. |
| `WINDUP_OUTPUT_DIR` | `~/temp/hill-of-towie-open-source-analysis/windup_output` | Root directory wind-up analyses write their per-assessment output into. |

Example `.env`:

```dotenv
HOT_OPEN_DATA_DIR=D:/hot_open/data
HOT_OPEN_CACHE_DIR=D:/hot_open/cache
WINDUP_OUTPUT_DIR=D:/hot_open/windup_output
```

## Contact

Alex.Clerc@res-group.com
