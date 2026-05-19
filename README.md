# Hill of Towie - Open Source Dataset

Uplift analysis of *Hill of Towie* wind farm using the open dataset: https://zenodo.org/records/20204946

The code within this repository contains analysis and helper functions to:
- download the [open dataset from Zenodo](https://zenodo.org/records/20204946)
- perform a wind farm energy analysis using the open dataset
- estimate the energy uplift using the [wind-up](https://github.com/resgroup/wind-up) library

The repository currently covers three upgrade validation campaigns:
- the **Dynamic Yaw** wake steering and collective yaw control trial, analysis first published in 2026 (`scripts/wake_steering_analysis`)
- the earlier **AeroUp** and **TuneUp** turbine upgrades, analysis first published in 2025 (`scripts/uplift_analysis_2025`)

## Quickstart

Analysis scripts in this repo download the input data from Zenodo automatically on first
use — no manual data staging is required. See `scripts/wake_steering_analysis/inspect_data.py`
for a short example of loading the open dataset into pandas dataframes.

## Dynamic Yaw analysis

The folder `scripts/wake_steering_analysis` analyses the energy uplift from the
**Dynamic Yaw (DY)** controller trial at Hill of Towie. DY combines two independent
contributions which are evaluated separately and then combined:

- **Wake Steering (WS)** — selected upwind turbines steer to
  deflect their wake away from a downwind partner. The controller applies a non-zero
  wake-steering offset only when the wind direction falls inside one of the pre-defined
  steering windows in `controller_config/wake-steering-lookup.csv`.
- **Collective Control (CC)** — every controlled turbine gets a better wind direction signal based on pooled information from all turbines.

DY is switched on and off by a synthetic toggle generated from UTC time (50-minute
on/off period, see `hot_dy_toggle_df()` in `uplift_per_steer.py`). The toggle pattern
lets wind-up estimate uplift from interleaved on/off pairs rather than a single
before/after split.

The driver script is `scripts/wake_steering_analysis/total_uplift.py`. It runs the WS
and CC analyses end-to-end, generates all the plots shown below, and combines the two
results into a long-term AEP figure via `compute_lt_uplift()`.

### Site layout

The bubble plot
below shows the layout and ground elevation of each turbine — the west cluster sits
on the higher, exposed hilltop while the east cluster drops down across a saddle:

<!-- TODO: upload `Hill of Towie turbine elevation.png` (from windup_output/HOT_dynamic_yaw_CC_only/plots) to GitHub and replace the URL below -->
![Hill of Towie turbine elevation](TODO_URL_turbine_elevation)

### Wake Steering result

`scripts/wake_steering_analysis/uplift_per_steer.py` runs one wind-up analysis per
steering window — each window uses a wind direction filter that matches the
upwind turbine's steering range. The P50 wake-steering uplift across the
qualifying steering events is **+1.32% (σ = 0.54%)**.

The plot below shows the per-turbine WS uplift across the west of the farm:

<!-- TODO: upload `Hill of Towie WS test turbine uplift.png` (from windup_output/HOT_dynamic_yaw/plots) -->
![Hill of Towie WS test turbine uplift](TODO_URL_ws_uplift)

### Collective Control result

`scripts/wake_steering_analysis/uplift_no_steering.py` runs wind-up on the 10-min
periods that the per-steer analysis did *not* consume, isolating the contribution of
CC to uplift. T07 is excluded from the CC test set because it was not actively controller during the campaign.

Across the 13 CC test turbines the P50 uplift is **+0.56% (σ = 0.47%)** with an
average yaw-activity reduction of **−6.0%**. The headline test-vs-reference bar
chart shows ~0% uplift on the four references (three referrence turbines and the ZX300 LiDAR) as expected, while the test
group is positive:

<!-- TODO: upload `Hill of Towie CC test turbine uplift.png` -->
![CC test turbine uplift](TODO_URL_cc_uplift)

<!-- TODO: upload `combined uplift and 90pct CI.png` -->
![Combined CC uplift and 90% CI](TODO_URL_cc_combined)

CC also changes how much each turbine yaws. The bubble plot below shows the
per-turbine yaw-activity change: blue means the turbine yawed less under CC (T10
shows the largest reduction at −14.6%):

<!-- TODO: upload `Hill of Towie CC test turbine yaw activity change.png` -->
![CC test turbine yaw activity change](TODO_URL_cc_yaw_change)

### Combined long-term AEP

`compute_lt_uplift()` in `total_uplift.py` blends the WS and CC results into a
long-term AEP figure using FLORIS-derived fractions of time and energy that the
long-term steering schedule will spend in each mode. The combination gives:

- **AEP uplift P50: +0.7%**, σ = 0.4%, P95 = 0.0%
- Long-term yaw-activity change: **−2.4%** wind farm average, with an individual
  turbine maximum of **+8.1%** on T03 (the most actively-steered turbine).

### `scripts/wake_steering_analysis` also includes:

- `inspect_data.py` — quick SCADA-vs-fastlog sanity plots.
- `zx300_wake_steers.py`, `zxtm_wake_steers.py` — wake steer finding plots from the
  on-site ZX300 (near T11) and ZXTM (on T07) LiDARs.
- `z300_wake_steer_animation.py`, `zxtm_wake_steer_animation.py` — matplotlib
  animations of LiDAR and turbine data during steering events.

## Earlier upgrade analysis (published in 2025)

The folder `scripts/uplift_analysis_2025` uses [wind-up](https://github.com/resgroup/wind-up) to run two analyses of energy uplift
after turbine upgrades.

Note any analysis can be run without running the others, e.g. you do not need to run `northing.py` before `aero_up.py`.

You can set an environment variable `WINDUP_ANALYSIS_DIR` to specify the location of the analysis directory.

The script `scripts/uplift_analysis_2025/northing.py` calculates the northing corrections saved to
`scripts/uplift_analysis_2025/wind_up_config/northing/optimized_northing_corrections.yaml` and used in subsequent analyses. The plot
below shows the circular difference of ERA5 wind direction to each turbine's yaw direction before the northing
correction. The turbine north calibrations are apparently wrong quite often which means the northing correction is quite
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

## contact

Alex.Clerc@res-group.com
