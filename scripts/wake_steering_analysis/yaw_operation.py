import logging
from pathlib import Path

import pandas as pd
from wake_steering_analysis.overall_uplift import CONFIG_DIR
from wind_up.models import WindUpConfig

from hot_open.settings import get_out_dir, get_wind_up_output_dir
from scripts.logger import setup_logger

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
    cfg.out_dir = get_wind_up_output_dir(cfg.assessment_name)

    pp_dir = cfg.out_dir / "pp_df"
    results = []
    for wtg_name in [x.name for x in cfg.asset.wtgs]:
        # TODO filter to used references
        pre_df = pd.DataFrame()
        for f in pp_dir.glob(f"{wtg_name}_*_pre_df.parquet"):
            if f.stem.split("_")[1] == wtg_name:
                continue
            _pre_df = pd.read_parquet(f)
            _pre_df = _pre_df.drop(columns=[x for x in _pre_df.columns if x.startswith("ref_")])
            pre_df = _pre_df if pre_df.empty else pre_df.combine_first(_pre_df).sort_index()
        post_df = pd.DataFrame()
        for f in pp_dir.glob(f"{wtg_name}_*_post_df.parquet"):
            if f.stem.split("_")[1] == wtg_name:
                continue
            _post_df = pd.read_parquet(f)
            _post_df = _post_df.drop(columns=[x for x in _post_df.columns if x.startswith("ref_")])
            post_df = _post_df if post_df.empty else post_df.combine_first(_post_df).sort_index()
        # TODO I also want "computed_core_post_processed_core_wake_steering_offset_degrees" but I need to gather the data from server and I need to resample to 10min prior to wind-up analysis
        pw_col = "test_ActivePowerMean"
        ya_col = "test_YawOperationCounts"
        pre_df = pre_df.dropna(subset=[pw_col, ya_col])
        post_df = post_df.dropna(subset=[pw_col, ya_col])

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
    pd.DataFrame(results).to_csv(out_dir / "results.csv")
