import logging
from pathlib import Path

import pandas as pd
from wind_up.models import WindUpConfig

from hot_open.settings import get_out_dir, get_wind_up_output_dir
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.hot_wake_steering_helpers import CONFIG_DIR

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    config_file_name = "HOT_dynamic_yaw.yaml"
    save_plots = True
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    uplift_per_steer_dir = get_wind_up_output_dir(cfg.assessment_name)

    results = []
    for wtg_name in [x.name for x in cfg.asset.wtgs]:
        pre_df = pd.DataFrame()
        post_df = pd.DataFrame()
        for dir in list(uplift_per_steer_dir.glob("T[0-9]*_T[0-9]*_*")):
            upwind_wtg_name = dir.stem.split("_")[0]
            downwind_wtg_name = dir.stem.split("_")[1]
            if not (wtg_name == upwind_wtg_name or wtg_name == downwind_wtg_name):
                continue
            ref_name = "_".join(dir.stem.split("_")[2:])
            pp_df_dir = dir / "pp_df"

            pre_df_path = pp_df_dir / f"{wtg_name}_{ref_name}_pre_df.parquet"
            post_df_path = pp_df_dir / f"{wtg_name}_{ref_name}_post_df.parquet"
            if not all([x.exists() for x in [pre_df_path, post_df_path]]):
                continue
            _pre_df = pd.read_parquet(pre_df_path)
            _pre_df = _pre_df.drop(columns=[x for x in _pre_df.columns if x.startswith("ref_")])
            pre_df = _pre_df if pre_df.empty else pre_df.combine_first(_pre_df).sort_index()
            _post_df = pd.read_parquet(post_df_path)
            _post_df = _post_df.drop(columns=[x for x in _post_df.columns if x.startswith("ref_")])
            post_df = _post_df if post_df.empty else post_df.combine_first(_post_df).sort_index()
        if pre_df.empty or post_df.empty:
            continue
        # TODO I also want "computed_core_post_processed_core_wake_steering_offset_degrees" but I need to gather the data from server and I need to resample to 10min prior to wind-up analysis
        pw_col = "test_ActivePowerMean"
        ya_col = "test_YawOperationCounts"
        toggle_col = "test_mean_toggle_state"
        control_active_col = "test_count_yaw_target_command_active"
        abs_wakesteer_col = "test_mean_abs_wake_steer_command"
        pre_df = pre_df.dropna(subset=[pw_col, ya_col, toggle_col, control_active_col])
        post_df = post_df.dropna(subset=[pw_col, ya_col, toggle_col, control_active_col])
        # measure yaw actions per hour and yaw actions per MWh pre and post
        timebase_s = 600
        hours_pre = len(pre_df) * timebase_s / 3600
        hours_post = len(post_df) * timebase_s / 3600
        mwh_pre = pre_df[pw_col].sum() / 1000 * timebase_s / 3600
        mwh_post = post_df[pw_col].sum() / 1000 * timebase_s / 3600
        ya_pre = pre_df[ya_col].sum()
        ya_post = post_df[ya_col].sum()
        yaph_pre = ya_pre / hours_pre
        yaph_post = ya_post / hours_post
        yapmwh_pre = ya_pre / mwh_pre
        yapmwh_post = ya_post / mwh_post
        results.append(
            {
                "wtg_name": wtg_name,
                "mean_toggle_col_pre": pre_df[toggle_col].mean(),
                "mean_toggle_col_post": post_df[toggle_col].mean(),
                "mean_control_active_col_pre": pre_df[control_active_col].mean(),
                "mean_control_active_col_post": post_df[control_active_col].mean(),
                "mean_abs_wakesteer_col_pre": pre_df[abs_wakesteer_col].mean(),
                "mean_abs_wakesteer_col_post": post_df[abs_wakesteer_col].mean(),
                "hours_pre": hours_pre,
                "hours_post": hours_post,
                "mwh_pre": mwh_pre,
                "mwh_post": mwh_post,
                "ya_pre": ya_pre,
                "ya_post": ya_post,
                "yaph_pre": yaph_pre,
                "yaph_post": yaph_post,
                "yapmwh_pre": yapmwh_pre,
                "yapmwh_post": yapmwh_post,
            }
        )
    pd.DataFrame(results).to_csv(out_dir / f"{Path(__file__).stem}_results.csv")
