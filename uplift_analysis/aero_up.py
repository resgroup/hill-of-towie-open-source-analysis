"""Analyse the uplift of the T13 AeroUp upgrade."""

import logging
from pathlib import Path

import pandas as pd
from wind_up.combine_results import combine_results
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

from hot_open import download_zenodo_data, setup_logger
from hot_open.paths import ANALYSES_DIR
from hot_open.unpack import unpack_local_meta_data, unpack_local_scada_data

CONFIG_DIR = Path(__file__).parent / "wind_up_config"
ANALYSIS_DIR = ANALYSES_DIR / Path(__file__).stem
ANALYSIS_CACHE_DIR = ANALYSIS_DIR / "cache"
ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _main_aeroup_analysis(
    *,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    analysis_output_dir: Path,
) -> None:
    setup_logger(analysis_output_dir / "analysis.log")
    logger = logging.getLogger(__name__)

    logger.info("Loading reference reanalysis data")
    reanalysis_file_path = Path(__file__).parent / "reanalysis_data/ERA5T_57.50N_-3.25E_100m_1hr_20241231.parquet"
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_57.50N_-3.25E_100m_1hr",
        data=pd.read_parquet(reanalysis_file_path),
    )

    logger.info("Defining Assessment Configuration")
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / "HoT_AeroUp_T13.yaml")
    cfg.out_dir = ANALYSIS_DIR / cfg.assessment_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = PlotConfig(show_plots=False, save_plots=True, plots_dir=cfg.out_dir / "plots")

    (ANALYSIS_CACHE_DIR / cfg.assessment_name).mkdir(parents=True, exist_ok=True)
    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df,
        metadata_df=metadata_df,
        reanalysis_datasets=[reanalysis_dataset],
        cache_dir=ANALYSIS_CACHE_DIR / cfg.assessment_name,
    )
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
    combined_results_df = combine_results(results_per_test_ref_df, plot_config=plot_cfg, auto_choose_refs=True)
    combined_results_df.to_csv(
        cfg.out_dir / f"{cfg.assessment_name}_combined_results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
    )


if __name__ == "__main__":
    setup_logger(ANALYSIS_DIR / f"{Path(__file__).stem}.log")
    download_zenodo_data(
        record_id="14870023",
        filenames=[
            *[f"{x}.zip" for x in range(2016, 2025)],
            "Hill_of_Towie_ShutdownDuration.zip",
            "Hill_of_Towie_turbine_metadata.csv",
        ],
    )
    metadata_df = unpack_local_meta_data()
    scada_df = unpack_local_scada_data()
    _main_aeroup_analysis(scada_df=scada_df, metadata_df=metadata_df, analysis_output_dir=ANALYSIS_DIR)
