from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from .utils import ROOT, ensure_dirs, save_csv


OUTCOME = "cumulative_excess_deaths_per_million_2020_2022"


def _load_data(root: Path) -> pd.DataFrame:
    scores = pd.read_csv(root / "results/tables/block_scores.csv")
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix_expanded.csv")
    consensus = pd.read_csv(root / "results/tables/consensus_rank_distribution.csv")
    n_ranked = float(consensus["n_countries_in_sample"].max())
    consensus["original_minimal_score"] = 1.0 - (consensus["median_rank"] - 1.0) / max(n_ranked - 1.0, 1.0)
    meta_cols = [
        c
        for c in [
            "iso3",
            "uhc_service_coverage_index",
            "current_health_expenditure_per_capita_ppp",
            "physicians_per_1000",
            "gdp_per_capita_ppp",
            "age65_share",
            "population_density",
        ]
        if c in matrix.columns
    ]
    data = scores.merge(outcomes[["iso3", OUTCOME]], on="iso3", how="inner")
    data = data.merge(matrix[meta_cols].drop_duplicates("iso3"), on="iso3", how="left", suffixes=("", "_raw"))
    data = data.merge(consensus[["iso3", "original_minimal_score"]], on="iso3", how="left")
    return data


def _cv_rmse(df: pd.DataFrame, feature_cols: list[str], model_name: str = "ridge") -> tuple[float, float, float]:
    work = df[feature_cols + [OUTCOME]].dropna().copy()
    if len(work) < 20:
        return np.nan, np.nan, np.nan
    x = work[feature_cols].astype(float)
    y = work[OUTCOME].astype(float).to_numpy()
    model = Ridge(alpha=10.0) if model_name == "ridge" else LinearRegression()
    if model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=20260624)
    cv = KFold(n_splits=min(5, max(2, len(work) // 20)), shuffle=True, random_state=20260624)
    pred = cross_val_predict(model, x, y, cv=cv)
    model.fit(x, y)
    in_pred = model.predict(x)
    p = len(feature_cols)
    r2 = r2_score(y, in_pred)
    adj = 1.0 - (1.0 - r2) * (len(work) - 1) / max(len(work) - p - 1, 1)
    return float(mean_squared_error(y, pred) ** 0.5), float(r2), float(adj)


def _stability(df: pd.DataFrame, risk_score: pd.Series, group_col: str) -> float:
    vals = []
    tmp = df[[OUTCOME, group_col]].copy()
    tmp["risk_score"] = risk_score
    for value in tmp[group_col].dropna().unique():
        sub = tmp[tmp[group_col].ne(value)].dropna()
        if len(sub) >= 20 and sub["risk_score"].nunique() > 2:
            vals.append(spearmanr(sub["risk_score"], sub[OUTCOME]).statistic)
    return float(np.nanstd(vals)) if vals else np.nan


def _evaluate_score(df: pd.DataFrame, label: str, risk_score: pd.Series, feature_cols: list[str] | None = None, model_name: str = "ridge") -> dict[str, object]:
    tmp = df[["iso3", "country", "region", "income_group", OUTCOME]].copy()
    tmp["risk_score"] = risk_score
    pair = tmp[["risk_score", OUTCOME]].dropna()
    n = len(pair)
    spearman = spearmanr(pair["risk_score"], pair[OUTCOME]).statistic if n >= 4 else np.nan
    kendall = kendalltau(pair["risk_score"], pair[OUTCOME]).statistic if n >= 4 else np.nan
    y_top = (tmp[OUTCOME] >= tmp[OUTCOME].quantile(0.75)).astype(int)
    auc = roc_auc_score(y_top[pd.notna(tmp["risk_score"])], tmp.loc[pd.notna(tmp["risk_score"]), "risk_score"]) if y_top.nunique() == 2 and tmp["risk_score"].notna().sum() > 10 else np.nan
    cv_rmse, r2, adj = _cv_rmse(df, feature_cols or ["risk_score"], model_name=model_name) if feature_cols else _cv_rmse(tmp, ["risk_score"], model_name="ridge")
    return {
        "model_or_benchmark": label,
        "n": n,
        "n_validation_countries": df["iso3"].nunique(),
        "missing_count": int(df["iso3"].nunique() - n),
        "missing_rate": float(1.0 - n / max(df["iso3"].nunique(), 1)),
        "spearman_with_shock_burden": float(spearman) if np.isfinite(spearman) else np.nan,
        "kendall_with_shock_burden": float(kendall) if np.isfinite(kendall) else np.nan,
        "cross_validated_rmse": cv_rmse,
        "adjusted_r2": adj,
        "top_risk_quartile_auc": float(auc) if np.isfinite(auc) else np.nan,
        "leave_one_region_spearman_std": _stability(tmp, tmp["risk_score"], "region"),
        "leave_one_income_group_spearman_std": _stability(tmp, tmp["risk_score"], "income_group"),
    }


def run_block_ablation(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    data = _load_data(root)
    data["log_gdp_per_capita_ppp"] = np.log(data["gdp_per_capita_ppp"].where(data["gdp_per_capita_ppp"] > 0))
    data["population_density_log"] = np.log1p(data["population_density"].clip(lower=0))
    variants = {
        "original_minimal_model": ["original_minimal_score"],
        "capacity_only": ["capacity_score"],
        "preparedness_only": ["preparedness_score"],
        "vulnerability_only": ["vulnerability_score"],
        "equity_access_only": ["equity_access_score"],
        "capacity_plus_preparedness": ["capacity_score", "preparedness_score"],
        "capacity_plus_vulnerability": ["capacity_score", "vulnerability_score"],
        "capacity_plus_equity_access": ["capacity_score", "equity_access_score"],
        "all_blocks": ["capacity_score", "preparedness_score", "vulnerability_score", "equity_access_score"],
    }
    rows = []
    for label, cols in variants.items():
        risk = -data[cols].mean(axis=1)
        rows.append(_evaluate_score(data, label, risk, feature_cols=cols))
    out = pd.DataFrame(rows)
    out["interpretation_note"] = "Higher risk-score values are expected to align with higher shock burden; descriptive predictive association only."
    save_csv(out, root / "results/tables/block_ablation_validation.csv")
    _plot_metric(out, "model_or_benchmark", "spearman_with_shock_burden", root / "results/figures/block_ablation_performance.pdf", "Block ablation performance")
    return out, data


def _pca_composite(data: pd.DataFrame, cols: list[str]) -> pd.Series:
    work = data[cols].astype(float)
    work = work.fillna(work.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(work)
    comp = PCA(n_components=1, random_state=20260624).fit_transform(scaled).ravel()
    corr = np.corrcoef(comp, data[cols].mean(axis=1).fillna(data[cols].mean(axis=1).median()))[0, 1]
    if corr < 0:
        comp = -comp
    return pd.Series(comp, index=data.index)


def run_benchmark_comparison(root: Path = ROOT) -> pd.DataFrame:
    data = _load_data(root)
    baseline = pd.read_csv(root / "results/tables/baseline_rankings.csv")
    pipeline = pd.read_csv(root / "results/tables/pipeline_grid_results.csv")
    rows = []
    block_cols = ["capacity_score", "preparedness_score", "vulnerability_score", "equity_access_score"]
    benchmarks: list[tuple[str, pd.Series, list[str] | None, str]] = [
        ("UHC service coverage index", -data["uhc_service_coverage_index"], ["uhc_service_coverage_index"], "ridge"),
        ("Health expenditure per capita", -data["current_health_expenditure_per_capita_ppp"], ["current_health_expenditure_per_capita_ppp"], "ridge"),
        ("Physicians per 1,000", -data["physicians_per_1000"], ["physicians_per_1000"], "ridge"),
        ("Equal-weight block composite", -data[block_cols].mean(axis=1), block_cols, "ridge"),
        ("PCA block composite", -_pca_composite(data, block_cols), None, "ridge"),
        ("Ridge regression predictor", pd.Series(np.nan, index=data.index), block_cols, "ridge"),
    ]
    if len(data) >= 80:
        benchmarks.append(("Random forest predictor", pd.Series(np.nan, index=data.index), block_cols, "random_forest"))

    # Existing single-pipeline benchmarks, joined by country.
    merec = baseline[baseline["ranking_method"].eq("marcos")].drop_duplicates("iso3")
    if not merec.empty:
        data = data.merge(merec[["iso3", "score"]].rename(columns={"score": "single_critic_marcos_score"}), on="iso3", how="left")
        benchmarks.append(("single CRITIC-MARCOS pipeline", -data["single_critic_marcos_score"], ["single_critic_marcos_score"], "ridge"))
    ent = pipeline[
        pipeline["weighting_method"].eq("entropy")
        & pipeline["ranking_method"].eq("topsis")
        & pipeline["imputation_method"].eq("knn")
        & pipeline["normalization_method"].eq("robust_minmax")
    ].drop_duplicates("iso3")
    if not ent.empty:
        data = data.merge(ent[["iso3", "score"]].rename(columns={"score": "entropy_topsis_score"}), on="iso3", how="left")
        benchmarks.append(("entropy-TOPSIS pipeline", -data["entropy_topsis_score"], ["entropy_topsis_score"], "ridge"))

    for label, risk, cols, model_name in benchmarks:
        if label in ["Ridge regression predictor", "Random forest predictor"]:
            cv_rmse, r2, adj = _cv_rmse(data, cols or block_cols, model_name=model_name)
            # Use in-sample fitted predictions only for association and AUC display.
            work = data[(cols or block_cols) + [OUTCOME]].dropna().copy()
            model = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=20260624) if model_name == "random_forest" else Ridge(alpha=10.0)
            model.fit(work[cols or block_cols], work[OUTCOME])
            pred = pd.Series(np.nan, index=data.index)
            pred.loc[work.index] = model.predict(work[cols or block_cols])
            row = _evaluate_score(data, label, pred, feature_cols=cols or block_cols, model_name=model_name)
            row["cross_validated_rmse"] = cv_rmse
            row["adjusted_r2"] = adj
        else:
            row = _evaluate_score(data, label, risk, feature_cols=cols, model_name=model_name)
        row["adds_hsr_convert_capability"] = "benchmark only; lacks full conversion-gap and bottleneck diagnostics"
        rows.append(row)
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/benchmark_comparison.csv")
    save_csv(out[["model_or_benchmark", "n", "missing_count", "missing_rate", "top_risk_quartile_auc"]], root / "results/tables/top_risk_auc_comparison.csv")
    save_csv(out[["model_or_benchmark", "n", "missing_count", "missing_rate", "cross_validated_rmse", "adjusted_r2"]], root / "results/tables/cross_validated_prediction_comparison.csv")
    _plot_metric(out, "model_or_benchmark", "spearman_with_shock_burden", root / "results/figures/benchmark_comparison.pdf", "Benchmark comparison")
    _plot_metric(out, "model_or_benchmark", "top_risk_quartile_auc", root / "results/figures/top_risk_auc_comparison.pdf", "Top-risk quartile AUC")
    return out


def _plot_metric(df: pd.DataFrame, label_col: str, metric_col: str, path: Path, title: str) -> None:
    plot = df.sort_values(metric_col)
    plt.figure(figsize=(9, max(5, 0.35 * len(plot))))
    plt.barh(plot[label_col], plot[metric_col], color="#2b8cbe")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel(metric_col.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def run_block_ablation_and_benchmarks(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ablation, _ = run_block_ablation(root)
    benchmarks = run_benchmark_comparison(root)
    return ablation, benchmarks


if __name__ == "__main__":
    run_block_ablation_and_benchmarks()
