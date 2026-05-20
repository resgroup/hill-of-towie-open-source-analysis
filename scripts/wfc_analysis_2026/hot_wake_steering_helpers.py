import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent / "wind_up_config"


_PW_COL = "test_ActivePowerMean"
_YA_COL = "test_YawOperationCounts"
_TOGGLE_COL = "test_mean_toggle_state"
_CONTROL_ACTIVE_COL = "test_count_yaw_target_command_active"
_ABS_WAKESTEER_COL = "test_mean_abs_wake_steer_command"
_TIMEBASE_S = 600


def calc_hot_yaw_stats(pre_df: pd.DataFrame, post_df: pd.DataFrame) -> dict:
    pre_df = pre_df.dropna(subset=[_PW_COL, _YA_COL, _TOGGLE_COL, _CONTROL_ACTIVE_COL])
    post_df = post_df.dropna(subset=[_PW_COL, _YA_COL, _TOGGLE_COL, _CONTROL_ACTIVE_COL])
    hours_pre = len(pre_df) * _TIMEBASE_S / 3600
    hours_post = len(post_df) * _TIMEBASE_S / 3600
    mwh_pre = pre_df[_PW_COL].sum() / 1000 * _TIMEBASE_S / 3600
    mwh_post = post_df[_PW_COL].sum() / 1000 * _TIMEBASE_S / 3600
    ya_pre = pre_df[_YA_COL].sum()
    ya_post = post_df[_YA_COL].sum()
    return {
        "mean_toggle_col_pre": pre_df[_TOGGLE_COL].mean(),
        "mean_toggle_col_post": post_df[_TOGGLE_COL].mean(),
        "mean_control_active_col_pre": pre_df[_CONTROL_ACTIVE_COL].mean(),
        "mean_control_active_col_post": post_df[_CONTROL_ACTIVE_COL].mean(),
        "mean_abs_wakesteer_col_pre": pre_df[_ABS_WAKESTEER_COL].mean()
        if _ABS_WAKESTEER_COL in pre_df.columns
        else np.nan,
        "mean_abs_wakesteer_col_post": post_df[_ABS_WAKESTEER_COL].mean()
        if _ABS_WAKESTEER_COL in post_df.columns
        else np.nan,
        "pp_hours_pre": hours_pre,
        "pp_hours_post": hours_post,
        "pp_mwh_pre": mwh_pre,
        "pp_mwh_post": mwh_post,
        "ya_pre": ya_pre,
        "ya_post": ya_post,
        "yaph_pre": ya_pre / hours_pre if hours_pre > 0 else np.nan,
        "yaph_post": ya_post / hours_post if hours_post > 0 else np.nan,
        "yapmwh_pre": ya_pre / mwh_pre if mwh_pre > 0 else np.nan,
        "yapmwh_post": ya_post / mwh_post if mwh_post > 0 else np.nan,
    }
