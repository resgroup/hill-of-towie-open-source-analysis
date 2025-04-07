# hill-of-towie-open-source-analysis
Project for open analysis of the Hill of Towie wind farm open dataset: https://zenodo.org/records/14870023

## Loading data
Example code for downloading data from Zenodo and loading it into pandas dataframes is shown in `uplift_analysis/01_northing.py`.

## Uplift analysis
The folder `uplift_analysis` uses [wind-up](https://github.com/resgroup/wind-up) to run two analyses of energy uplift after turbine upgrades.

The script `uplift_analysis/01_northing.py` calculates the northing corrections saved to `uplift_analysis/wind_up_config/northing/optimized_northing_corrections.yaml` and used in subsequent analyses. The plot below shows the circular difference of ERA5 wind direction to each turbine's yaw direction before the northing correction. The turbine north calibrations are apparently wrong quite often which means the northing correction is quite important.
![northing error vs reanalysis_wd before northing](https://github.com/user-attachments/assets/aaf1e4c6-dc10-4c59-9281-1051128464af)

The script `uplift_analysis/02_aero_up.py` analyses the energy uplift thanks to [AeroUp](https://www.res-group.com/digital-solutions/aeroup/) for T13. The result is a P50 uplift of 4.3% with a 90% confidence interval of 3.3% to 5.3%.

The script `uplift_analysis/03_tune_up.py` analyses the energy uplift thanks to [TuneUp](https://www.res-group.com/digital-solutions/tuneup/) for nine test turbines. The result is a P50 uplift of 1.1% with a 90% confidence interval of 0.2% to 2.0%.

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
