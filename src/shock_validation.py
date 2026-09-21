from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .plotting import plot_leave_region_out, plot_resilience_vs_outcome
from .utils import ROOT, ensure_dirs, save_csv


OUTCOME = "cumulative_excess_deaths_per_million_2020_2022"


def _bootstrap_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    method: str,
    n_boot: int = 1000,
    seed: int = 20260624,
) -> tuple[float, float, float, float]:
    x = df[x_col].astype(float)
    y = df[y_col].astype(float)
    if method == "spearman":
        stat, pval = spearmanr(x, y, nan_policy="omit")
    elif method == "kendall":
        stat, pval = kendalltau(x, y, nan_policy="omit")
    else:
        raise ValueError(method)
    rng = np.random.default_rng(seed)
    boots = []
    values = df[[x_col, y_col]].dropna().reset_index(drop=True)
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = values.iloc[idx]
        if sample[x_col].nunique() < 2 or sample[y_col].nunique() < 2:
            continue
        if method == "spearman":
            boots.append(spearmanr(sample[x_col], sample[y_col]).statistic)
        else:
            boots.append(kendalltau(sample[x_col], sample[y_col]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(stat), float(lo), float(hi), float(pval)


def _model_pipeline(df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    categorical = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    numeric = [c for c in feature_cols if c not in categorical]
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", transformer), ("model", LinearRegression())])


def _fit_regression(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    n_boot: int = 500,
    seed: int = 20260624,
) -> dict[str, object]:
    work = df[feature_cols + [OUTCOME]].dropna().copy()
    if len(work) < len(feature_cols) + 5:
        return {
            "model": model_name,
            "n": len(work),
            "controls": ",".join([c for c in feature_cols if c != "resilience_score_rank_based"]),
            "resilience_coef": np.nan,
            "coef_ci_low": np.nan,
            "coef_ci_high": np.nan,
            "r2": np.nan,
            "adjusted_r2": np.nan,
            "status": "insufficient sample size",
        }

    x = work[feature_cols]
    y = work[OUTCOME].astype(float)
    pipe = _model_pipeline(work, feature_cols)
    pipe.fit(x, y)
    pred = pipe.predict(x)
    r2 = r2_score(y, pred)
    p = pipe.named_steps["prep"].transform(x).shape[1]
    adjusted = 1.0 - (1.0 - r2) * (len(work) - 1) / max(len(work) - p - 1, 1)

    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_
    target_name = [name for name in names if name.endswith("resilience_score_rank_based")]
    coef = float(coefs[list(names).index(target_name[0])]) if target_name else np.nan

    rng = np.random.default_rng(seed)
    boot_coefs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(work), len(work))
        sample = work.iloc[idx]
        if sample[OUTCOME].nunique() < 2 or sample["resilience_score_rank_based"].nunique() < 2:
            continue
        try:
            bpipe = _model_pipeline(sample, feature_cols)
            bpipe.fit(sample[feature_cols], sample[OUTCOME].astype(float))
            bnames = bpipe.named_steps["prep"].get_feature_names_out()
            btarget = [name for name in bnames if name.endswith("resilience_score_rank_based")]
            if btarget:
                boot_coefs.append(float(bpipe.named_steps["model"].coef_[list(bnames).index(btarget[0])]))
        except Exception:
            continue
    ci_low, ci_high = np.nanpercentile(boot_coefs, [2.5, 97.5]) if boot_coefs else (np.nan, np.nan)

    return {
        "model": model_name,
        "n": len(work),
        "controls": ",".join([c for c in feature_cols if c != "resilience_score_rank_based"]) or "none",
        "resilience_coef": coef,
        "coef_ci_low": float(ci_low),
        "coef_ci_high": float(ci_high),
        "r2": float(r2),
        "adjusted_r2": float(adjusted),
        "status": "fit",
    }


def _fit_huber(df: pd.DataFrame) -> dict[str, object]:
    cols = ["resilience_score_rank_based", "log_gdp_per_capita_ppp", "age65_share", "population_density_log"]
    work = df[cols + [OUTCOME]].dropna().copy()
    if len(work) < 20:
        return {"model": "huber_basic", "n": len(work), "status": "insufficient sample size"}
    x = StandardScaler().fit_transform(work[cols])
    y = work[OUTCOME].astype(float).to_numpy()
    model = HuberRegressor().fit(x, y)
    pred = model.predict(x)
    return {
        "model": "huber_basic",
        "n": len(work),
        "controls": "log_gdp_per_capita_ppp,age65_share,population_density_log",
        "resilience_coef": float(model.coef_[0]),
        "coef_ci_low": np.nan,
        "coef_ci_high": np.nan,
        "r2": float(r2_score(y, pred)),
        "adjusted_r2": np.nan,
        "status": "fit",
    }


def run_shock_validation(root: Path = ROOT) -> pd.DataFrame:
    ensure_dirs(root)
    consensus = pd.read_csv(root / "results/tables/consensus_rank_distribution.csv")
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix.csv")
    controls = matrix[
        [
            "iso3",
            "gdp_per_capita_ppp",
            "age65_share",
            "population_density",
            "life_expectancy",
            "region",
            "income_group",
        ]
    ].copy()
    validation = consensus.merge(outcomes[["iso3", "outcome_date", OUTCOME]], on="iso3", how="inner")
    validation = validation.merge(controls, on="iso3", how="left", suffixes=("", "_control"))
    n_countries = validation["n_countries_in_sample"].max()
    validation["resilience_score_rank_based"] = 1.0 - (validation["median_rank"] - 1.0) / max(float(n_countries) - 1.0, 1.0)
    validation["log_gdp_per_capita_ppp"] = np.log(validation["gdp_per_capita_ppp"].where(validation["gdp_per_capita_ppp"] > 0))
    validation["population_density_log"] = np.log1p(validation["population_density"].clip(lower=0))
    save_csv(validation, root / "results/processed_outputs/shock_validation_dataset.csv")
    n_ranked_countries = len(consensus)
    n_validation_countries = len(validation)
    outcome_missing_count = n_ranked_countries - n_validation_countries
    outcome_missing_rate = outcome_missing_count / max(n_ranked_countries, 1)

    corr_rows = []
    for x_col, label in [
        ("resilience_score_rank_based", "rank_based_resilience_score"),
        ("median_rank", "median_rank"),
        ("top_quartile_probability", "top_quartile_probability"),
    ]:
        for method in ["spearman", "kendall"]:
            stat, lo, hi, pval = _bootstrap_corr(validation, x_col, OUTCOME, method)
            corr_rows.append(
                {
                    "x_variable": label,
                    "outcome": OUTCOME,
                    "method": method,
                    "n": int(validation[[x_col, OUTCOME]].dropna().shape[0]),
                    "n_ranked_countries": n_ranked_countries,
                    "outcome_missing_count_from_ranked_sample": outcome_missing_count,
                    "outcome_missing_rate_from_ranked_sample": outcome_missing_rate,
                    "statistic": stat,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": pval,
                    "interpretation_note": "Association only; outcome was not used as an input feature.",
                }
            )
    correlations = pd.DataFrame(corr_rows)
    save_csv(correlations, root / "results/tables/shock_validation_correlations.csv")

    regression_specs = [
        ("score_only", ["resilience_score_rank_based"]),
        ("plus_gdp", ["resilience_score_rank_based", "log_gdp_per_capita_ppp"]),
        (
            "plus_demographics",
            ["resilience_score_rank_based", "log_gdp_per_capita_ppp", "age65_share", "population_density_log"],
        ),
        (
            "plus_region_income",
            [
                "resilience_score_rank_based",
                "log_gdp_per_capita_ppp",
                "age65_share",
                "population_density_log",
                "region",
                "income_group",
            ],
        ),
    ]
    regression_rows = [_fit_regression(validation, cols, name) for name, cols in regression_specs]
    regression_rows.append(_fit_huber(validation))
    regressions = pd.DataFrame(regression_rows)
    regressions["n_ranked_countries"] = n_ranked_countries
    regressions["n_validation_countries"] = n_validation_countries
    regressions["outcome_missing_count_from_ranked_sample"] = outcome_missing_count
    regressions["outcome_missing_rate_from_ranked_sample"] = outcome_missing_rate
    save_csv(regressions, root / "results/tables/shock_validation_regressions.csv")

    q1 = validation["median_rank"].quantile(0.25)
    q3 = validation["median_rank"].quantile(0.75)
    top = validation[validation["median_rank"].le(q1)][OUTCOME].dropna()
    bottom = validation[validation["median_rank"].ge(q3)][OUTCOME].dropna()
    if len(top) and len(bottom):
        test = mannwhitneyu(top, bottom, alternative="two-sided")
        quartile = pd.DataFrame(
            [
                {
                    "comparison": "top_resilience_quartile_vs_bottom_resilience_quartile",
                    "n_ranked_countries": n_ranked_countries,
                    "n_validation_countries": n_validation_countries,
                    "outcome_missing_count_from_ranked_sample": outcome_missing_count,
                    "outcome_missing_rate_from_ranked_sample": outcome_missing_rate,
                    "top_quartile_n": len(top),
                    "bottom_quartile_n": len(bottom),
                    "top_quartile_median_outcome": float(top.median()),
                    "bottom_quartile_median_outcome": float(bottom.median()),
                    "median_difference_top_minus_bottom": float(top.median() - bottom.median()),
                    "mann_whitney_u": float(test.statistic),
                    "p_value": float(test.pvalue),
                }
            ]
        )
    else:
        quartile = pd.DataFrame(
            [
                {
                    "comparison": "top_resilience_quartile_vs_bottom_resilience_quartile",
                    "n_ranked_countries": n_ranked_countries,
                    "n_validation_countries": n_validation_countries,
                    "outcome_missing_count_from_ranked_sample": outcome_missing_count,
                    "outcome_missing_rate_from_ranked_sample": outcome_missing_rate,
                    "top_quartile_n": len(top),
                    "bottom_quartile_n": len(bottom),
                    "top_quartile_median_outcome": np.nan,
                    "bottom_quartile_median_outcome": np.nan,
                    "median_difference_top_minus_bottom": np.nan,
                    "mann_whitney_u": np.nan,
                    "p_value": np.nan,
                }
            ]
        )
    save_csv(quartile, root / "results/tables/quartile_outcome_comparison.csv")

    leave_rows = []
    for region in sorted(validation["region"].dropna().unique()):
        held = validation[validation["region"].ne(region)]
        if len(held) >= 20 and held["resilience_score_rank_based"].nunique() > 2:
            stat = spearmanr(held["resilience_score_rank_based"], held[OUTCOME], nan_policy="omit").statistic
            excluded_n = int(validation["region"].eq(region).sum())
            leave_rows.append(
                {
                    "held_out_region": region,
                    "n": len(held),
                    "excluded_region_n": excluded_n,
                    "n_ranked_countries": n_ranked_countries,
                    "n_validation_countries": n_validation_countries,
                    "outcome_missing_count_from_ranked_sample": outcome_missing_count,
                    "outcome_missing_rate_from_ranked_sample": outcome_missing_rate,
                    "spearman_r": float(stat),
                }
            )
    leave_region = pd.DataFrame(leave_rows)
    save_csv(leave_region, root / "results/tables/leave_region_out_validation.csv")

    plot_resilience_vs_outcome(validation, root / "results/figures/resilience_vs_excess_mortality.pdf")
    plot_leave_region_out(leave_region, root / "results/figures/leave_region_out_validation.pdf")
    return validation


if __name__ == "__main__":
    run_shock_validation()
