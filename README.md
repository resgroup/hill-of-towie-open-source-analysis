# Hill of Towie - Open Source Dataset

Uplift analysis of *Hill of Towie* wind farm using the open dataset: https://zenodo.org/records/14870023

The code within this repository contains analysis and helper functions to:
- download the [open dataset from Zenodo](https://zenodo.org/records/14870023)
- perform a wind farm energy analysis using the open dataset
- estimate the energy uplift using the [wind-up](https://github.com/resgroup/wind-up) library

## Quickstart

Example code for downloading data from Zenodo and loading it into pandas dataframes is shown in
`uplift_analysis/northing.py`.

The scripts in `uplift_analysis` create many plots of the data including a plot of the 
layout and capacity factor of each turbine as shown below:

![Hill of Towie capacity factor](https://github.com/user-attachments/assets/915473d0-3871-4758-adc8-534ab1cd8acc)


## Uplift analysis

The folder `uplift_analysis` uses [wind-up](https://github.com/resgroup/wind-up) to run two analyses of energy uplift
after turbine upgrades.

Note any analysis can be run without running the others, e.g. you do not need to run `northing.py` before `aero_up.py`.

The script `uplift_analysis/northing.py` calculates the northing corrections saved to
`uplift_analysis/wind_up_config/northing/optimized_northing_corrections.yaml` and used in subsequent analyses. The plot
below shows the circular difference of ERA5 wind direction to each turbine's yaw direction before the northing
correction. The turbine north calibrations are apparently wrong quite often which means the northing correction is quite
important.

![northing error vs reanalysis_wd before northing](https://github.com/user-attachments/assets/aaf1e4c6-dc10-4c59-9281-1051128464af)

### AeroUp

The script `uplift_analysis/aero_up.py` analyses the energy uplift thanks
to [AeroUp](https://www.res-group.com/digital-solutions/aeroup/) for T13.
The result is a P50 uplift of 4.3% with a 90% confidence interval of 3.3% to 5.3%. This is visualized in the plot below
along with the uplift results for the three selected reference turbines, which are near 0% uplift as expected:

![combined uplift and 90% CI](https://github.com/user-attachments/assets/a36214d1-7308-4a16-9be3-9e2230170708)


### TuneUp

The script `uplift_analysis/tune_up.py` analyses the energy uplift thanks
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
