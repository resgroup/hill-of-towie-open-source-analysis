import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from wind_up.combine_results import _calc_sigma_ref
from wind_up.models import PlotConfig
from wind_up.plots.combine_results_plots import plot_testref_and_combined_results

from scripts.wfc_analysis_2026.hot_wake_steering_helpers import calc_hot_yaw_stats

logger = logging.getLogger(__name__)


def _load_yaw_stats(row, wind_up_out_dir: Path) -> dict | None:
    pp_df_dir = wind_up_out_dir / "pp_df"
    pre_path = pp_df_dir / f"{row.test_wtg}_{row.ref}_pre_df.parquet"
    post_path = pp_df_dir / f"{row.test_wtg}_{row.ref}_post_df.parquet"
    if not pre_path.exists() or not post_path.exists():
        logger.warning("pp_df files not found for %s (ref %s)", row.test_wtg, row.ref)
        return None
    logger.info("Reading: %s", pre_path)
    pre_df = pd.read_parquet(pre_path)
    logger.info("Reading: %s", post_path)
    post_df = pd.read_parquet(post_path)
    return calc_hot_yaw_stats(pre_df, post_df)


def _calc_cc_only_tdf(
    trdf: pd.DataFrame, ref_list: list[str], weight_col: str = "unc_weight", sigma_ref: float = 0
) -> pd.DataFrame:
    tdf = trdf.groupby("test_wtg").agg(
        p50_uplift=pd.NamedAgg(
            column="uplift_frc",
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


def combine_cc_only_results(
    per_turbine_results,
    *,
    auto_choose_refs: bool = False,
    exclude_refs: list[str] | None = None,
    plot_config: PlotConfig | None = None,
) -> pd.DataFrame:
    """This is a copy of wind-up combine_cc_only_results except for use of _calc_cc_only_tdf"""
    if exclude_refs is None:
        exclude_refs = []

    msg = "#" * 78 + "\n# combine results per test turbine\n" + "#" * 78
    logger.info(msg)

    trdf = per_turbine_results.copy()

    if trdf.groupby(["test_wtg", "ref"]).size().max() > 1:
        msg = "trdf must have no more than one row per test-ref pair"
        raise ValueError(msg)

    # remove reference predictions of themselves
    trdf = trdf.loc[trdf["test_wtg"] != trdf["ref"], :]

    if len(exclude_refs) > 0:
        logger.info(f"excluding refs {exclude_refs}")
        trdf = trdf.loc[~trdf["test_wtg"].isin(exclude_refs), :]
        trdf = trdf.loc[~trdf["ref"].isin(exclude_refs), :]

    if (trdf["unc_one_sigma_frc"] <= 0).any() or trdf["unc_one_sigma_frc"].isna().any():
        msg = "unc_one_sigma_frc must be positive and non-NaN"
        raise ValueError(msg)

    weight_col = "unc_weight"
    trdf[weight_col] = 1 / (trdf["unc_one_sigma_frc"] ** 2)

    ref_list = sorted(trdf["ref"].unique())

    min_refs = 3
    if auto_choose_refs:
        if len(ref_list) >= min_refs:
            best_ref_list = _choose_best_refs(trdf, ref_list, min_refs=min_refs)  # noqa: F821
            refs_to_remove = [x for x in ref_list if x not in best_ref_list]
            trdf = trdf.loc[~trdf["test_wtg"].isin(refs_to_remove), :]
            trdf = trdf.loc[~trdf["ref"].isin(refs_to_remove), :]
            ref_list = sorted(trdf["ref"].unique())
        else:
            result_manager.warning(f"len(ref_list) < {min_refs}, skipping auto_choose_refs")  # noqa: F821

    logger.info(f"ref_list = {ref_list}")
    tdf = _calc_cc_only_tdf(trdf, ref_list, weight_col)

    # change column order for readability
    cols = list(tdf.columns)
    first_cols = ["test_wtg", "p50_uplift", "p95_uplift", "p5_uplift", "sigma"]
    cols = first_cols + [x for x in cols if x not in first_cols]
    tdf = tdf[cols]

    if plot_config is not None:
        plot_testref_and_combined_results(trdf=trdf, tdf=tdf, plot_cfg=plot_config)

    return tdf


def combine_cc_results_with_yaw(
    per_turbine_results,
    *,
    wind_up_out_dir: Path,
    plot_config: PlotConfig | None = None,
    exclude_refs: list[str] | None = None,
) -> pd.DataFrame:
    yaw_stats_rows = {}
    for row in per_turbine_results.itertuples():
        stats = _load_yaw_stats(row, wind_up_out_dir)
        yaw_stats_rows[row.Index] = stats if stats is not None else {}
    yaw_stats_df = pd.DataFrame.from_dict(yaw_stats_rows, orient="index")
    per_turbine_results = per_turbine_results.join(yaw_stats_df)
    per_turbine_with_yaw_path = wind_up_out_dir / "HOT_dynamic_yaw_CC_only_results_per_test_ref_with_yaw.csv"
    logger.info("Writing: %s", per_turbine_with_yaw_path)
    per_turbine_results.to_csv(per_turbine_with_yaw_path, index=False)
    tdf = combine_cc_only_results(per_turbine_results, plot_config=plot_config, exclude_refs=exclude_refs)
    tdf["yaph_change"] = (tdf["yaph_post"] - tdf["yaph_pre"]) / tdf["yaph_pre"]
    return tdf
