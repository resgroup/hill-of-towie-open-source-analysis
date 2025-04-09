"""Analyse the uplift of the TuneUp upgrades."""

import getpass
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import pandas as pd
from wind_up.combine_results import calculate_total_uplift_of_test_and_ref_turbines, combine_results
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.reanalysis_data import ReanalysisDataset

from hot_open import setup_logger
from hot_open.unpack import unpack_local_meta_data, unpack_local_scada_data

OUT_DIR = Path.home() / "hill-of-towie-open-source-analysis" / Path(__file__).stem
CACHE_DIR = OUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR = Path(__file__).parent / "wind_up_config"

logger = logging.getLogger(__name__)


def _main_tuneup_analysis(
    *,
    config_fname: str,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    analysis_output_dir: Path,
) -> None:
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    setup_logger(analysis_output_dir / f"{config_fname.split('.')[0]}_analysis.log")
    logger = logging.getLogger(__name__)

    logger.info("Loading reference reanalysis data")
    reanalysis_file_path = Path(__file__).parent / "reanalysis_data/ERA5T_57.50N_-3.25E_100m_1hr_20241231.parquet"
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_57.50N_-3.25E_100m_1hr",
        data=pd.read_parquet(reanalysis_file_path),
    )

    logger.info("Defining Assessment Configuration")
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_fname)
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


@dataclass
class HoTPitchTuneUpResults:
    """Class representing test-ref results and settings for a HoT pitch TuneUp zone."""

    config_file_name: str
    results_per_test_ref_fname: str
    refs: list[str]
    test_wtgs_to_exclude: list[str] | None = None


if __name__ == "__main__":
    re_run_test_ref_results = False
    if re_run_test_ref_results:
        username = getpass.getuser()

        data_dir = OUT_DIR.parent / "zenodo_data"

        metadata_df = unpack_local_meta_data(data_dir=data_dir)

        scada_df = unpack_local_scada_data(data_dir=data_dir)
        # The wind farm is analysed in three sections to avoid an excessive number of test-ref combinations
        for config_fname in [
            "HoT_PitchTuneUp2024_north.yaml",
            "HoT_PitchTuneUp2024_east.yaml",
            "HoT_PitchTuneUp2024_south.yaml",
        ]:
            _main_tuneup_analysis(
                config_fname=config_fname,
                scada_df=scada_df,
                metadata_df=metadata_df,
                analysis_output_dir=OUT_DIR,
            )

    show_plots = False
    save_plots = True

    all_combined_results_df = pd.DataFrame()
    for result in [
        HoTPitchTuneUpResults(
            config_file_name="HoT_PitchTuneUp2024_east.yaml",
            results_per_test_ref_fname="HoT_PitchTuneUp2024_east_results_per_test_ref_20250328_232321.csv",
            refs=["T17", "T18", "T19"],
        ),
        HoTPitchTuneUpResults(
            config_file_name="HoT_PitchTuneUp2024_south.yaml",
            results_per_test_ref_fname="HoT_PitchTuneUp2024_south_results_per_test_ref_20250329_030039.csv",
            refs=["T05", "T01", "T04", "T07"],
        ),
        HoTPitchTuneUpResults(
            config_file_name="HoT_PitchTuneUp2024_north.yaml",
            results_per_test_ref_fname="HoT_PitchTuneUp2024_north_results_per_test_ref_20250328_220318.csv",
            refs=["T11", "T12", "T14"],
            test_wtgs_to_exclude=[
                "T08",  # T08 south results are used since they have lower uncertainty
            ],
        ),
    ]:
        cfg = WindUpConfig.from_yaml(CONFIG_DIR / result.config_file_name)
        cfg.out_dir = OUT_DIR / cfg.assessment_name
        plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")

        msg = f"{cfg.assessment_name=}"
        logger.info(msg)

        results_per_test_ref_df = pd.read_csv(cfg.out_dir / result.results_per_test_ref_fname, index_col=0)
        ref_list_before = list(results_per_test_ref_df["ref"].unique())
        msg = f"{ref_list_before=}"
        logger.info(msg)
        refs_to_exclude = [x for x in ref_list_before if x not in result.refs]
        msg = f"{refs_to_exclude=}"
        logger.info(msg)
        combined_results_df = combine_results(
            results_per_test_ref_df, plot_config=plot_cfg, auto_choose_refs=False, exclude_refs=refs_to_exclude
        )
        combined_results_df.to_csv(
            cfg.out_dir
            / f"{cfg.assessment_name}_combined_results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv",
        )

        # role is ref if test_wtg is in result.refs else test
        combined_results_df["role"] = combined_results_df["test_wtg"].apply(
            partial(lambda x, result: "ref" if x in result.refs else "test", result=result)
        )
        combined_results_df["assessment_name"] = cfg.assessment_name
        # remove exclude rows
        if result.test_wtgs_to_exclude is not None:
            combined_results_df = combined_results_df[
                ~combined_results_df["test_wtg"].isin(result.test_wtgs_to_exclude)
            ]

        all_combined_results_df = pd.concat([all_combined_results_df, combined_results_df]).reset_index(drop=True)

    plot_cfg.plots_dir = plot_cfg.plots_dir.parent.parent
    wf_results = calculate_total_uplift_of_test_and_ref_turbines(all_combined_results_df, plot_cfg=plot_cfg)
    wf_results.to_csv(OUT_DIR / f"wf_results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
