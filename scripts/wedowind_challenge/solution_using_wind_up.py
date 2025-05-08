"""Make example submission file for the WeDoWind challenge using wind-up.

Challenge URL: https://www.kaggle.com/competitions/hill-of-towie-wind-turbine-power-prediction
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from wind_up.constants import REANALYSIS_WD_COL, REANALYSIS_WS_COL, WINDFARM_YAWDIR_COL, DataColumns
from wind_up.detrend import apply_wsratio_v_wd_scen, calc_wsratio_v_wd_scen
from wind_up.interface import AssessmentInputs
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.northing import check_wtg_northing
from wind_up.plots.data_coverage_plots import plot_detrend_data_cov
from wind_up.plots.detrend_plots import plot_apply_wsratio_v_wd_scen
from wind_up.reanalysis_data import ReanalysisDataset
from wind_up.waking_state import add_waking_scen, get_distance_and_bearing
from wind_up.windspeed_drift import check_windspeed_drift

from hot_open import download_zenodo_data, setup_logger
from hot_open.paths import ANALYSES_DIR
from hot_open.unpack import unpack_local_meta_data, unpack_local_scada_data

load_dotenv()

ANALYSIS_DIR = ANALYSES_DIR / Path(__file__).stem
ANALYSIS_CACHE_DIR = ANALYSIS_DIR / "cache"
ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


def wind_up_features_for_kaggle(
    *,
    scada_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    analysis_output_dir: Path,
) -> pd.DataFrame:
    """Make a dataframe of useful features for the HoT kaggle competition."""
    setup_logger(analysis_output_dir / "analysis.log")
    logger = logging.getLogger(__name__)

    uplift_analysis_dir = Path(__file__).parents[2] / "uplift_analysis"
    logger.info("Loading reference reanalysis data")
    reanalysis_file_path = uplift_analysis_dir / "reanalysis_data/ERA5T_57.50N_-3.25E_100m_1hr_20241231.parquet"
    reanalysis_dataset = ReanalysisDataset(
        id="ERA5T_57.50N_-3.25E_100m_1hr",
        data=pd.read_parquet(reanalysis_file_path),
    )

    logger.info("Defining Assessment Configuration")
    cfg = WindUpConfig.from_yaml(uplift_analysis_dir / "wind_up_config/HoT_wedowind_T1.yaml")
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
    wf_df = assessment_inputs.wf_df
    cfg = assessment_inputs.cfg
    plot_cfg = assessment_inputs.plot_cfg

    # the below code is adapted from the wind-up source code starting from
    # https://github.com/resgroup/wind-up/blob/d8ab91f3c656fa10c0986539683e626759c6ce4c/wind_up/main_analysis.py#L802
    test_name = "T01"
    test_wtg = next(x for x in cfg.asset.wtgs if x.name == test_name)
    test_pw_col = "pw_clipped" if cfg.clip_rated_power_pp else DataColumns.active_power_mean
    test_ws_col = "ws_est_from_power_only" if cfg.ignore_turbine_anemometer_data else "ws_est_blend"
    test_df = wf_df.loc[test_wtg.name].copy()

    lt_df_raw = None
    lt_df_filt = None

    test_df.columns = ["test_" + x for x in test_df.columns]
    test_pw_col = "test_" + test_pw_col
    test_ws_col = "test_" + test_ws_col

    check_windspeed_drift(
        wtg_df=test_df,
        wtg_name=test_name,
        ws_col=test_ws_col,
        reanalysis_ws_col="test_" + REANALYSIS_WS_COL,
        cfg=cfg,
        plot_cfg=plot_cfg,
    )

    test_df, _, _ = assessment_inputs.pre_post_splitter.split(test_df, test_wtg_name=test_name)

    # create an emp
    predicted_power_df = pd.DataFrame(index=test_df.index)

    for ref_wtg in cfg.ref_wtgs:
        ref_name = ref_wtg.name
        (plot_cfg.plots_dir / test_name / ref_name).mkdir(exist_ok=True, parents=True)
        if test_name == ref_name:
            ref_ws_col = DataColumns.wind_speed_mean
            test_ws_col = "test_" + DataColumns.wind_speed_mean
        else:
            ref_ws_col = "ws_est_from_power_only" if cfg.ignore_turbine_anemometer_data else "ws_est_blend"
        ref_info = {
            "ref": ref_name,
        }
        ref_wd_col = "YawAngleMean"
        ref_df = wf_df.loc[ref_name].copy()
        ref_max_northing_error_v_reanalysis = check_wtg_northing(
            ref_df,
            wtg_name=ref_name,
            north_ref_wd_col=REANALYSIS_WD_COL,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            sub_dir=f"{test_name}/{ref_name}",
        )
        ref_max_northing_error_v_wf = check_wtg_northing(
            ref_df,
            wtg_name=ref_name,
            north_ref_wd_col=WINDFARM_YAWDIR_COL,
            timebase_s=cfg.timebase_s,
            plot_cfg=plot_cfg,
            sub_dir=f"{test_name}/{ref_name}",
        )

        ref_ws_col = "ref_" + ref_ws_col
        ref_wd_col = "ref_" + ref_wd_col
        ref_df.columns = ["ref_" + x for x in ref_df.columns]

        test_lat = test_wtg.latitude
        test_long = test_wtg.longitude
        ref_lat = ref_wtg.latitude
        ref_long = ref_wtg.longitude

        distance_m, bearing_deg = get_distance_and_bearing(
            lat1=test_lat,
            long1=test_long,
            lat2=ref_lat,
            long2=ref_long,
        )

        ref_max_ws_drift, ref_max_ws_drift_pp_period = check_windspeed_drift(
            wtg_df=ref_df,
            wtg_name=ref_name,
            ws_col=ref_ws_col,
            reanalysis_ws_col="ref_" + REANALYSIS_WS_COL,
            cfg=cfg,
            plot_cfg=plot_cfg,
            sub_dir=f"{test_name}/{ref_name}",
        )

        detrend_df = test_df.merge(ref_df, how="left", left_index=True, right_index=True)
        detrend_df = detrend_df[cfg.detrend_first_dt_utc_start : cfg.detrend_last_dt_utc_start]  # type: ignore[misc]

        detrend_df = add_waking_scen(
            test_name=test_name,
            ref_name=ref_name,
            test_ref_df=detrend_df,
            cfg=cfg,
            wf_df=wf_df,
            ref_wd_col=ref_wd_col,
            ref_lat=ref_lat,
            ref_long=ref_long,
        )

        plot_detrend_data_cov(
            cfg=cfg,
            test_name=test_name,
            ref_name=ref_name,
            test_df=test_df,
            test_ws_col=test_ws_col,
            ref_df=ref_df,
            ref_ws_col=ref_ws_col,
            ref_wd_col=ref_wd_col,
            detrend_df=detrend_df,
            plot_cfg=plot_cfg,
        )

        wsratio_v_dir_scen = calc_wsratio_v_wd_scen(
            test_name=test_name,
            ref_name=ref_name,
            ref_lat=ref_lat,
            ref_long=ref_long,
            detrend_df=detrend_df,
            test_ws_col=test_ws_col,
            ref_ws_col=ref_ws_col,
            ref_wd_col=ref_wd_col,
            cfg=cfg,
            plot_cfg=plot_cfg,
        )

        out_df = test_df.merge(ref_df, how="left", left_index=True, right_index=True)

        out_df = add_waking_scen(
            test_ref_df=out_df,
            test_name=test_name,
            ref_name=ref_name,
            cfg=cfg,
            wf_df=wf_df,
            ref_wd_col=ref_wd_col,
            ref_lat=ref_lat,
            ref_long=ref_long,
        )

        detrend_ws_col = "ref_ws_detrended"
        out_df = apply_wsratio_v_wd_scen(out_df, wsratio_v_dir_scen, ref_ws_col=ref_ws_col, ref_wd_col=ref_wd_col)
        plot_apply_wsratio_v_wd_scen(
            out_df.dropna(subset=[ref_ws_col, test_ws_col, detrend_ws_col, test_pw_col]),
            ref_ws_col=ref_ws_col,
            test_ws_col=test_ws_col,
            detrend_ws_col=detrend_ws_col,
            test_pw_col=test_pw_col,
            test_name=test_name,
            ref_name=ref_name,
            title_end="out_df",
            plot_cfg=plot_cfg,
        )

        # predict T1 power
        detrend_ws_col = "ref_ws_detrended"
        ref_number = int(ref_name.replace("T", ""))
        predicted_power_df[f"ref_ws_detrended;{ref_number}"] = out_df[detrend_ws_col]
        predicted_power_df[f"power_prediction;{ref_number}"] = pd.Series(
            np.interp(
                out_df[detrend_ws_col],
                assessment_inputs.pc_per_ttype["SWT-2.3-82"]["WindSpeedMean"].to_numpy(),
                assessment_inputs.pc_per_ttype["SWT-2.3-82"]["pw_clipped"].to_numpy(),
            ),
            index=out_df.index,
        )
    return predicted_power_df


if __name__ == "__main__":
    setup_logger(ANALYSIS_DIR / f"{Path(__file__).stem}.log")
    download_zenodo_data(
        record_id="14870023",
        filenames=[
            *[f"{x}.zip" for x in range(2016, 2020 + 1)],
            "Hill_of_Towie_ShutdownDuration.zip",
            "Hill_of_Towie_turbine_metadata.csv",
        ],
    )
    metadata_df = unpack_local_meta_data()
    scada_df = unpack_local_scada_data(end_dt_excl=pd.Timestamp("2021-01-01", tz="UTC"))
    predicted_power_df = wind_up_features_for_kaggle(
        scada_df=scada_df, metadata_df=metadata_df, analysis_output_dir=ANALYSIS_DIR
    )
    # prefer closest turbines
    predicted_power_df["prediction"] = predicted_power_df[["T02", "T03", "T04"]].mean(axis=1)
    # try nan rows again with all turbines
    na_rows = predicted_power_df["prediction"].isna()
    predicted_power_df.loc[na_rows, "prediction"] = predicted_power_df.loc[
        na_rows, ["T02", "T03", "T04", "T05", "T07"]
    ].mean(axis=1)
    # fill as a last resort
    predicted_power_df["prediction"] = predicted_power_df["prediction"].interpolate().ffill().bfill()

    # fill as a last resort
    submission_example = pd.read_parquet(
        Path.home() / "Downloads" / "hill-of-towie-wind-turbine-power-prediction" / "submission_dataset.parquet"
    ).set_index("TimeStamp_StartFormat")
    submission_df = (
        submission_example[["id"]]
        .merge(
            predicted_power_df[["prediction"]],
            how="left",
            left_index=True,
            right_index=True,
        )
        .set_index("id")
        .sort_index()
    )
    output_fpath = ANALYSIS_DIR / "kaggle_submission.csv"
    submission_df.to_csv(output_fpath)
    msg = f"Submission file saved to {output_fpath}"
    logger.info(msg)

    # Optional: checking the submission file
    _df = pd.read_csv(output_fpath)

    # checking the columns are the expected ones
    assert _df.columns.to_list() == ["id", "prediction"], (  # noqa:S101
        f'Expected columns ["id", "prediction"], found: {_df.columns.to_list()}'
    )

    # checking no nulls in the data
    assert _df.isna().sum().sum() == 0, "There are NA values in the data!"  # noqa:S101

    # checking the row ids are unique and within expected range
    duplicated_ids = _df["id"].duplicated()
    assert not duplicated_ids.any(), f"There are duplicated ids: {_df['id'][duplicated_ids].to_numpy()}"  # noqa:S101
    invalid_ids = set(_df["id"].unique()) - set(range(52704))
    assert not invalid_ids, f"The following row IDs are not within the expected ones: {invalid_ids}"  # noqa:S101
