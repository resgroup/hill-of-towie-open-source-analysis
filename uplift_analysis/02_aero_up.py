"""Analyse the uplift of the T13 AeroUp upgrade."""

import getpass
import logging
from pathlib import Path

import pandas as pd
from helpers import setup_logger
from input_data import unpack_local_meta_data, unpack_local_scada_data
from wind_up.combine_results import combine_results
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

OUT_DIR = Path.home() / "temp" / "hill-of-towie-open-source-analysis-internal" / Path(__file__).stem
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = Path(__file__).parent / "wind_up_config"


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
    cfg.out_dir = OUT_DIR / cfg.assessment_name
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = PlotConfig(show_plots=False, save_plots=False, plots_dir=cfg.out_dir / "plots")

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
    combined_results_df = combine_results(results_per_test_ref_df, plot_config=plot_cfg, auto_choose_refs=True)
    combined_results_df.to_csv(
        cfg.out_dir / f"{cfg.assessment_name}_combined_results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
    )


if __name__ == "__main__":
    username = getpass.getuser()
    data_dir = Path(r"C:\Users") / username / "RES Group/Digital Solutions - HardTech - Open source dataset"

    metadata_df = unpack_local_meta_data(data_dir=data_dir)

    scada_df = unpack_local_scada_data(data_dir=data_dir)
    _main_aeroup_analysis(
        scada_df=scada_df,
        metadata_df=metadata_df,
        analysis_output_dir=OUT_DIR,
    )
