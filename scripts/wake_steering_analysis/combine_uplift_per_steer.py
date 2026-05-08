import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from wind_up.models import PlotConfig, WindUpConfig

from hot_open.settings import get_out_dir, get_wind_up_output_dir
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.uplift_per_steer import CONFIG_DIR

logger = logging.getLogger(__name__)


def _calc_wakesteer_tdf(
    trdf: pd.DataFrame, ref_list: list[str], weight_col: str = "unc_weight", sigma_ref: float = 0
) -> pd.DataFrame:
    tdf = trdf.groupby(["upwind_wtg", "downwind_wtg"]).agg(
        p50_net_uplift=pd.NamedAgg(
            column="net_p50_uplift",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        hours_off=pd.NamedAgg(
            column="hours_off",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        hours_on=pd.NamedAgg(
            column="hours_on",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        upwind_uplift_p50=pd.NamedAgg(
            column="upwind_uplift_p50",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        downwind_uplift_p50=pd.NamedAgg(
            column="downwind_uplift_p50",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        first_wdir=pd.NamedAgg(
            column="first_wdir",
            aggfunc="median",
        ),
        last_wdir=pd.NamedAgg(
            column="last_wdir",
            aggfunc="median",
        ),
        sigma_uncorr=pd.NamedAgg(
            column="unc_one_sigma_frc",
            aggfunc=lambda x: np.sqrt(
                ((x * trdf.loc[x.index, weight_col] / trdf.loc[x.index, weight_col].sum()) ** 2).sum(),
            ),
        ),
        sigma_corr=pd.NamedAgg(
            column="unc_one_sigma_frc",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        ref_count=pd.NamedAgg(column="net_p50_uplift", aggfunc=len),
        ref_list=pd.NamedAgg(column="reference", aggfunc=lambda x: ", ".join(sorted(x))),
    )
    tdf["sigma_test"] = (tdf["sigma_uncorr"] + tdf["sigma_corr"]) / 2
    tdf = tdf.sort_values(by=["ref_count", "upwind_wtg", "downwind_wtg"], ascending=[False, True, True])
    tdf = tdf.reset_index()
    tdf["sigma"] = tdf["sigma_test"].clip(lower=sigma_ref)
    tdf["p95_net_uplift"] = tdf["p50_net_uplift"] + norm.ppf(0.05) * tdf["sigma"]
    tdf["p5_net_uplift"] = tdf["p50_net_uplift"] + norm.ppf(0.95) * tdf["sigma"]
    return tdf


def combine_wakesteer_results() -> pd.DataFrame:
    trdf = all_wakesteer_results.copy()
    trdf["unc_one_sigma_frc"] = (trdf["net_p5_uplift"] - trdf["net_p95_uplift"]) / 2 / norm.ppf(0.95)
    weight_col = "unc_weight"
    trdf[weight_col] = 1 / (trdf["unc_one_sigma_frc"] ** 2)
    ref_list = sorted(trdf["reference"].unique())
    return _calc_wakesteer_tdf(trdf, ref_list, weight_col)


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
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")

    all_wakesteer_results = pd.read_csv(cfg.out_dir / "uplift_per_steer_results.csv")
    combined_results = combine_wakesteer_results(all_wakesteer_results)
    combined_results.to_csv(cfg.out_dir / "uplift_per_steer_combined_results.csv")
