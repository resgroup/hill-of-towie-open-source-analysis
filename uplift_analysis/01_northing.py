"""Run northing analysis on the entire open source dataset.

This outputs optimized_northing_corrections.yaml which can be used as northing config for other analyses
"""

import logging
from pathlib import Path

import pandas as pd
from helpers import download_zenodo_data, setup_logger
from input_data import unpack_local_meta_data, unpack_local_scada_data
from wind_up.combine_results import combine_results
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

OUT_DIR = Path.home() / "temp" / "hill-of-towie-open-source-analysis" / Path(__file__).stem
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = Path(__file__).parent / "wind_up_config"
logger = logging.getLogger(__name__)


def _main_northing_analysis(
    *,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> None:
    logger.info("Loading reference reanalysis data")
    reanalysis_file_path = Path(__file__).parent / "reanalysis_data/ERA5T_57.50N_-3.25E_100m_1hr_20241231.parquet"
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_57.50N_-3.25E_100m_1hr",
        data=pd.read_parquet(reanalysis_file_path),
    )

    logger.info("Defining Assessment Configuration")
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / "HoT_northing.yaml")
    cfg.out_dir = OUT_DIR / cfg.assessment_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")

    (CACHE_DIR / cfg.assessment_name).mkdir(parents=True, exist_ok=True)
    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df,
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=CACHE_DIR / cfg.assessment_name,
    )
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
    combined_results_df = combine_results(results_per_test_ref_df, plot_config=plot_cfg, auto_choose_refs=False)
    combined_results_df.to_csv(
        cfg.out_dir / f"{cfg.assessment_name}_combined_results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
    )


if __name__ == "__main__":
    setup_logger(OUT_DIR / f"{Path(__file__).stem}.log")

    data_dir = OUT_DIR.parent / "zenodo_data"
    download_zenodo_data(
        record_id="14870023",
        output_dir=data_dir,
        filenames=[
            *[f"{x}.zip" for x in range(2016, 2025)],
            "Hill_of_Towie_ShutdownDuration.zip",
            "Hill_of_Towie_turbine_metadata.csv",
        ],
    )

    metadata_df = unpack_local_meta_data(data_dir=data_dir)
    scada_df = unpack_local_scada_data(data_dir=data_dir)
    _main_northing_analysis(
        scada_df=scada_df,
        metadata_df=metadata_df,
    )
