import logging
from functools import reduce
from pathlib import Path

import pandas as pd
from wind_up.combine_results import calculate_total_uplift_of_test_and_ref_turbines
from wind_up.interface import AssessmentInputs
from wind_up.main_analysis import run_wind_up_analysis
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.scada_funcs_plots import bubble_plot
from wind_up.smart_data import add_smart_lat_long_to_cfg

from hot_open.era5_helpers import get_hot_reanalysis_datasets
from hot_open.settings import get_cache_dir, get_out_dir, get_wind_up_output_dir
from hot_open.unpack import unpack_local_meta_data
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.combine_uplift_no_steering import combine_cc_results_with_yaw
from scripts.wake_steering_analysis.hot_wake_steering_helpers import CONFIG_DIR
from scripts.wake_steering_analysis.inspect_data import LOCAL_TEMPORARY_DIR
from scripts.wake_steering_analysis.uplift_per_steer import _hot_dy_lidar_datasets, hot_dy_scada_df, hot_dy_toggle_df

logger = logging.getLogger(__name__)


def _post_process_cc_results(*, wf_cc_results: pd.DataFrame, combined_cc_results_df) -> tuple[float, float, float]:
    cc_uplift = wf_cc_results.loc["test", "p50_uplift"]
    cc_uplift_uncertainty = wf_cc_results.loc["test", "sigma"]
    cc_yaph_change = combined_cc_results_df[~combined_cc_results_df["is_ref"]]["yaph_change"].mean()
    msg = f"CC test results P50 {100 * cc_uplift:.2f}%, uncertainty {100 * cc_uplift_uncertainty:.2f}%"
    logger.info(msg)
    msg = f"CC ref results P50 {100 * wf_cc_results.loc['ref', 'p50_uplift']:.2f}%"
    logger.info(msg)
    msg = f"CC test results yaw activity change {100 * cc_yaph_change:.1f}%"
    logger.info(msg)
    return cc_uplift, cc_uplift_uncertainty, cc_yaph_change


def hot_dy_uplift_no_steering(rerun_windup: bool = True) -> tuple[float, float, float]:
    config_file_name = "HOT_dynamic_yaw.yaml"
    save_plots = True
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    uplift_per_steer_dir = get_wind_up_output_dir(cfg.assessment_name)
    cfg.assessment_name = "HOT_dynamic_yaw_CC_only"
    cfg.out_dir = get_wind_up_output_dir(cfg.assessment_name)
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")

    if not rerun_windup:
        wf_cc_results = pd.read_csv(cfg.out_dir / "wf_cc_results.csv", index_col=0)
        combined_cc_results_df = pd.read_csv(cfg.out_dir / f"{cfg.assessment_name}_combined_cc_results.csv")
        cc_uplift, cc_uplift_uncertainty, cc_yaph_change = _post_process_cc_results(
            wf_cc_results=wf_cc_results, combined_cc_results_df=combined_cc_results_df
        )

        return cc_uplift, cc_uplift_uncertainty, cc_yaph_change

    scada_df = hot_dy_scada_df()
    scada_df["exclude_row"] = 0
    total_excluded = 0
    test_wtgs = [x.name for x in cfg.test_wtgs]
    refs = [x.name for x in cfg.ref_wtgs] + cfg.non_wtg_ref_names
    for dir in list(uplift_per_steer_dir.glob("T[0-9]*_T[0-9]*_*")):
        upwind_wtg_name = dir.stem.split("_")[0]
        downwind_wtg_name = dir.stem.split("_")[1]
        ref_name = "_".join(dir.stem.split("_")[2:])
        if (upwind_wtg_name not in test_wtgs) or (downwind_wtg_name not in test_wtgs) or (ref_name not in refs):
            continue
        pp_df_dir = dir / "pp_df"
        paths = [
            pp_df_dir / f"{upwind_wtg_name}_{ref_name}_pre_df.parquet",
            pp_df_dir / f"{downwind_wtg_name}_{ref_name}_pre_df.parquet",
            pp_df_dir / f"{upwind_wtg_name}_{ref_name}_post_df.parquet",
            pp_df_dir / f"{downwind_wtg_name}_{ref_name}_post_df.parquet",
        ]
        dfs = [pd.read_parquet(p) for p in paths if p.exists()]
        if len(dfs) == 0:
            continue
        combined_index = reduce(lambda a, b: a.union(b), (df.index for df in dfs))
        filter_data = scada_df.index.isin(combined_index) & scada_df["TurbineName"].isin(
            [upwind_wtg_name, downwind_wtg_name]
        )
        newly_excluded = int((filter_data & (scada_df["exclude_row"] == 0)).sum())
        scada_df.loc[filter_data, "exclude_row"] = 1
        total_excluded += newly_excluded
        logger.info("%s: %d rows newly excluded, running total %d", dir.stem, newly_excluded, total_excluded)

    assessment_inputs = AssessmentInputs.from_cfg(
        cfg=cfg,
        plot_cfg=plot_cfg,
        scada_df=scada_df,
        metadata_df=unpack_local_meta_data(data_dir=LOCAL_TEMPORARY_DIR, scada_index_name=scada_df.index.name),
        toggle_df=hot_dy_toggle_df(scada_df),
        reanalysis_datasets=get_hot_reanalysis_datasets(),
        cache_dir=get_cache_dir() / "windup_cache" / cfg.assessment_name,
        mast_or_lidar_datasets=_hot_dy_lidar_datasets(
            data_dir=LOCAL_TEMPORARY_DIR / "lidar_data",
            start_dt=scada_df.index.min(),
            end_dt_excl=scada_df.index.max(),
        ),
    )
    results_per_test_ref_df = run_wind_up_analysis(assessment_inputs)
    combined_cc_results_df = combine_cc_results_with_yaw(
        results_per_test_ref_df, wind_up_out_dir=cfg.out_dir, plot_config=plot_cfg
    )
    combined_cc_results_df.to_csv(
        cfg.out_dir / f"{cfg.assessment_name}_combined_cc_results.csv",
    )
    wf_cc_results = calculate_total_uplift_of_test_and_ref_turbines(combined_cc_results_df, plot_cfg=plot_cfg)
    wf_cc_results.to_csv(cfg.out_dir / "wf_cc_results.csv")

    cfg = add_smart_lat_long_to_cfg(md=unpack_local_meta_data(), cfg=cfg)
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")
    (cfg.out_dir / "plots").mkdir(parents=True, exist_ok=True)

    hot_elevations = pd.read_csv(Path(__file__).parent / "Hill of Towie_elevations.csv", index_col=0)
    title = f"{cfg.asset.name} turbine elevation"
    bubble_plot(
        cfg=cfg,
        series=hot_elevations["Elevation"],
        title=title,
        cbarunits="m",
        save_path=plot_cfg.plots_dir / f"{title}.png",
        show_plot=plot_cfg.show_plots,
    )
    title = f"{cfg.asset.name} CC test turbine yaw activity change"
    bubble_plot(
        cfg=cfg,
        series=combined_cc_results_df[~combined_cc_results_df["is_ref"]]
        .set_index("test_wtg")["yaph_change"]
        .sort_index()
        * 100,
        title=title,
        cbarunits="%",
        save_path=plot_cfg.plots_dir / f"{title}.png",
        show_plot=plot_cfg.show_plots,
    )
    title = f"{cfg.asset.name} CC test turbine uplift"
    bubble_plot(
        cfg=cfg,
        series=combined_cc_results_df[~combined_cc_results_df["is_ref"]]
        .set_index("test_wtg")["p50_uplift"]
        .sort_index()
        * 100,
        title=title,
        cbarunits="%",
        save_path=plot_cfg.plots_dir / f"{title}.png",
        show_plot=plot_cfg.show_plots,
    )

    cc_uplift, cc_uplift_uncertainty, cc_yaph_change = _post_process_cc_results(
        wf_cc_results=wf_cc_results, combined_cc_results_df=combined_cc_results_df
    )
    return cc_uplift, cc_uplift_uncertainty, cc_yaph_change


if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)
    hot_dy_uplift_no_steering()
