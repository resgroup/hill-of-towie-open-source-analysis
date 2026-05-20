import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.wfc_analysis_2026.hot_wake_steering_helpers import calc_hot_yaw_stats

logger = logging.getLogger(__name__)


def _load_yaw_stats_for_steer(row, wind_up_out_dir: Path) -> dict | None:
    steer_dir = wind_up_out_dir / f"{row.upwind_wtg}_{row.downwind_wtg}_{row.reference}"
    pp_df_dir = steer_dir / "pp_df"
    pre_path = pp_df_dir / f"{row.upwind_wtg}_{row.reference}_pre_df.parquet"
    post_path = pp_df_dir / f"{row.upwind_wtg}_{row.reference}_post_df.parquet"
    if not pre_path.exists() or not post_path.exists():
        logger.warning("pp_df files not found for %s -> %s (ref %s)", row.upwind_wtg, row.downwind_wtg, row.reference)
        return None
    logger.info("Reading: %s", pre_path)
    pre_df = pd.read_parquet(pre_path)
    logger.info("Reading: %s", post_path)
    post_df = pd.read_parquet(post_path)
    return calc_hot_yaw_stats(pre_df, post_df)


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
        mean_toggle_col_pre=pd.NamedAgg(
            column="mean_toggle_col_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        mean_toggle_col_post=pd.NamedAgg(
            column="mean_toggle_col_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        mean_control_active_col_pre=pd.NamedAgg(
            column="mean_control_active_col_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        mean_control_active_col_post=pd.NamedAgg(
            column="mean_control_active_col_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        mean_abs_wakesteer_col_pre=pd.NamedAgg(
            column="mean_abs_wakesteer_col_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        mean_abs_wakesteer_col_post=pd.NamedAgg(
            column="mean_abs_wakesteer_col_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        pp_hours_pre=pd.NamedAgg(
            column="pp_hours_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        pp_hours_post=pd.NamedAgg(
            column="pp_hours_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        pp_mwh_pre=pd.NamedAgg(
            column="pp_mwh_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        pp_mwh_post=pd.NamedAgg(
            column="pp_mwh_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        ya_pre=pd.NamedAgg(
            column="ya_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        ya_post=pd.NamedAgg(
            column="ya_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        yaph_pre=pd.NamedAgg(
            column="yaph_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        yaph_post=pd.NamedAgg(
            column="yaph_post",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        yapmwh_pre=pd.NamedAgg(
            column="yapmwh_pre",
            aggfunc=lambda x: (x * trdf.loc[x.index, weight_col]).sum() / trdf.loc[x.index, weight_col].sum(),
        ),
        yapmwh_post=pd.NamedAgg(
            column="yapmwh_post",
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


def combine_wakesteer_results(all_wakesteer_results, exclude_refs: list[str] | None = None) -> pd.DataFrame:
    if exclude_refs is None:
        exclude_refs = []
    trdf = all_wakesteer_results.copy()
    if len(exclude_refs) > 0:
        logger.info(f"excluding refs {exclude_refs}")
        trdf = trdf.loc[~trdf["reference"].isin(exclude_refs), :]
    trdf["unc_one_sigma_frc"] = (trdf["net_p5_uplift"] - trdf["net_p95_uplift"]) / 2 / norm.ppf(0.95)
    weight_col = "unc_weight"
    trdf[weight_col] = 1 / (trdf["unc_one_sigma_frc"] ** 2)
    ref_list = sorted(trdf["reference"].unique())
    tdf = _calc_wakesteer_tdf(trdf, ref_list, weight_col)
    tdf["yaph_change"] = (tdf["yaph_post"] - tdf["yaph_pre"]) / tdf["yaph_pre"]
    return tdf


def combine_wakesteer_results_with_yaw(
    all_wakesteer_results, *, wind_up_out_dir: Path, exclude_refs: list[str] | None = None
):
    yaw_stats_rows = {}
    for row in all_wakesteer_results.itertuples():
        stats = _load_yaw_stats_for_steer(row, wind_up_out_dir)
        yaw_stats_rows[row.Index] = stats if stats is not None else {}
    yaw_stats_df = pd.DataFrame.from_dict(yaw_stats_rows, orient="index")
    all_wakesteer_results = all_wakesteer_results.join(yaw_stats_df)
    results_with_yaw_path = wind_up_out_dir / "uplift_per_steer_results_with_yaw.csv"
    logger.info("Writing: %s", results_with_yaw_path)
    all_wakesteer_results.to_csv(results_with_yaw_path, index=False)
    return combine_wakesteer_results(all_wakesteer_results, exclude_refs=exclude_refs)
