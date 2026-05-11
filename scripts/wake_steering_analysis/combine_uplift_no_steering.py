import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from wind_up.combine_results import _calc_sigma_ref
from wind_up.models import PlotConfig, WindUpConfig
from wind_up.plots.combine_results_plots import plot_testref_and_combined_results

from hot_open.settings import get_out_dir, get_wind_up_output_dir
from scripts.logger import setup_logger
from scripts.wake_steering_analysis.hot_wake_steering_helpers import CONFIG_DIR, _calc_yaw_stats

logger = logging.getLogger(__name__)


def _load_yaw_stats(row, wind_up_out_dir: Path) -> dict | None:
    pp_df_dir = wind_up_out_dir / "pp_df"
    pre_path = pp_df_dir / f"{row.test_wtg}_{row.ref}_pre_df.parquet"
    post_path = pp_df_dir / f"{row.test_wtg}_{row.ref}_post_df.parquet"
    if not pre_path.exists() or not post_path.exists():
        logger.warning("pp_df files not found for %s (ref %s)", row.test_wtg, row.ref)
        return None
    pre_df = pd.read_parquet(pre_path)
    post_df = pd.read_parquet(post_path)
    return _calc_yaw_stats(pre_df, post_df)


def _calc_cc_only_tdf(
    trdf: pd.DataFrame, ref_list: list[str], weight_col: str = "unc_weight", sigma_ref: float = 0
) -> pd.DataFrame:
    tdf = trdf.groupby("test_wtg").agg(
        p50_uplift=pd.NamedAgg(
            column="uplift_frc",
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
        mean_toggle_col_pre=pd.NamedAgg(
            column="mean_toggle_col_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        mean_toggle_col_post=pd.NamedAgg(
            column="mean_toggle_col_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        mean_control_active_col_pre=pd.NamedAgg(
            column="mean_control_active_col_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        mean_control_active_col_post=pd.NamedAgg(
            column="mean_control_active_col_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        mean_abs_wakesteer_col_pre=pd.NamedAgg(
            column="mean_abs_wakesteer_col_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        mean_abs_wakesteer_col_post=pd.NamedAgg(
            column="mean_abs_wakesteer_col_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        pp_hours_pre=pd.NamedAgg(
            column="pp_hours_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        pp_hours_post=pd.NamedAgg(
            column="pp_hours_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        pp_mwh_pre=pd.NamedAgg(
            column="pp_mwh_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        pp_mwh_post=pd.NamedAgg(
            column="pp_mwh_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        ya_pre=pd.NamedAgg(
            column="ya_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        ya_post=pd.NamedAgg(
            column="ya_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        yaph_pre=pd.NamedAgg(
            column="yaph_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        yaph_post=pd.NamedAgg(
            column="yaph_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        yapmwh_pre=pd.NamedAgg(
            column="yapmwh_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
        ),
        yapmwh_post=pd.NamedAgg(
            column="yapmwh_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum()
            / trdf.loc[x.index, weight_col][x.notna()].sum(),
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
        ref_count=pd.NamedAgg(column="uplift_frc", aggfunc=len),
        ref_list=pd.NamedAgg(column="ref", aggfunc=lambda x: ", ".join(sorted(x))),
        is_ref=pd.NamedAgg(column="test_wtg", aggfunc=lambda x: x.isin(ref_list).any()),
    )
    tdf["sigma_test"] = (tdf["sigma_uncorr"] + tdf["sigma_corr"]) / 2
    tdf = tdf.sort_values(by=["ref_count", "test_wtg"], ascending=[False, True])
    tdf = tdf.reset_index()
    sigma_ref = _calc_sigma_ref(tdf, ref_list)
    tdf["sigma_ref"] = sigma_ref
    tdf["sigma"] = tdf["sigma_test"].clip(lower=sigma_ref)
    tdf["p95_uplift"] = tdf["p50_uplift"] + norm.ppf(0.05) * tdf["sigma"]
    tdf["p5_uplift"] = tdf["p50_uplift"] + norm.ppf(0.95) * tdf["sigma"]
    return tdf


def combine_cc_only_results(per_turbine_results, *, plot_config: PlotConfig | None = None) -> pd.DataFrame:
    trdf = per_turbine_results.copy()
    weight_col = "unc_weight"
    trdf[weight_col] = 1 / (trdf["unc_one_sigma_frc"] ** 2)
    ref_list = sorted(trdf["reference"].unique())
    tdf = _calc_cc_only_tdf(trdf, ref_list, weight_col)
    if plot_config is not None:
        plot_testref_and_combined_results(trdf=trdf, tdf=tdf, plot_cfg=plot_config)
    return tdf


def combine_cc_results_with_yaw(
    per_turbine_results, wind_up_out_dir: Path, *, plot_config: PlotConfig | None = None
) -> pd.DataFrame:
    yaw_stats_rows = {}
    for row in per_turbine_results.itertuples():
        stats = _load_yaw_stats(row, wind_up_out_dir)
        yaw_stats_rows[row.Index] = stats if stats is not None else {}
    yaw_stats_df = pd.DataFrame.from_dict(yaw_stats_rows, orient="index")
    per_turbine_results = per_turbine_results.join(yaw_stats_df)
    per_turbine_results.to_csv(
        wind_up_out_dir / "HOT_dynamic_yaw_CC_only_results_per_test_ref_with_yaw.csv", index=False
    )
    logger.info("saved HOT_dynamic_yaw_CC_only_results_per_test_ref_with_yaw.csv")
    return combine_cc_only_results(per_turbine_results, plot_config=plot_config)


if __name__ == "__main__":
    override_wind_up_output_dir = Path(
        r"C:\Users\aclerc\temp\hill-of-towie-open-source-analysis\server windup_output\HOT_dynamic_yaw_CC_only"
    )

    out_dir = get_out_dir(dir_name=Path(__file__).stem)
    log_path = out_dir / f"{Path(__file__).stem}.log"
    setup_logger(log_path)
    msg = f"log file is at {log_path}"
    logger.info(msg)

    config_file_name = "HOT_dynamic_yaw.yaml"
    save_plots = True
    cfg = WindUpConfig.from_yaml(CONFIG_DIR / config_file_name)
    cfg.assessment_name = "HOT_dynamic_yaw_CC_only"
    cfg.out_dir = (
        get_wind_up_output_dir(cfg.assessment_name)
        if override_wind_up_output_dir is None
        else override_wind_up_output_dir
    )
    plot_cfg = PlotConfig(show_plots=False, save_plots=save_plots, plots_dir=cfg.out_dir / "plots")

    per_turbine_results = pd.read_csv(cfg.out_dir / "HOT_dynamic_yaw_CC_only_results_per_test_ref_20260509_081651.csv")
    combined_results_df = combine_cc_results_with_yaw(per_turbine_results, cfg.out_dir)
    combined_results_df.to_csv(cfg.out_dir / "HOT_dynamic_yaw_CC_combined_results_with_yaw.csv")
