from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import ROOT, ensure_dirs, save_csv


OUTCOME = "cumulative_excess_deaths_per_million_2020_2022"
PROFILE_LABELS = [
    "effective converters",
    "capacity under-realizers",
    "adaptive over-performers",
    "structurally vulnerable systems",
    "uncertain / data-limited systems",
]


def _potential_capacity_table(root: Path) -> pd.DataFrame:
    block_scores = pd.read_csv(root / "results/tables/block_scores.csv")
    dist = pd.read_csv(root / "results/tables/block_rank_distributions.csv")
    full = dist[dist["block"].eq("full_potential_capacity")].copy()
    cols = [
        "iso3",
        "median_rank",
        "rank_p05",
        "rank_p95",
        "top_quartile_probability",
        "bottom_quartile_probability",
        "method_disagreement_index",
        "n_valid_pipelines",
        "n_countries_in_sample",
        "n_indicators",
    ]
    potential = block_scores.merge(full[cols], on="iso3", how="left", suffixes=("", "_ensemble"))
    potential = potential.rename(
        columns={
            "full_potential_capacity_score": "potential_capacity_score",
            "full_potential_capacity_rank": "potential_capacity_rank_point",
            "median_rank": "median_potential_capacity_rank",
            "rank_p05": "potential_rank_p05",
            "rank_p95": "potential_rank_p95",
        }
    )
    out_cols = [
        "iso3",
        "country",
        "region",
        "income_group",
        "potential_capacity_score",
        "potential_capacity_rank_point",
        "median_potential_capacity_rank",
        "potential_rank_p05",
        "potential_rank_p95",
        "top_quartile_probability",
        "bottom_quartile_probability",
        "method_disagreement_index",
        "capacity_score",
        "preparedness_score",
        "vulnerability_score",
        "equity_access_score",
        "expanded_feature_missing_count",
        "expanded_feature_missing_rate",
        "n_valid_pipelines",
        "n_countries_in_sample",
        "n_indicators",
    ]
    out = potential[[c for c in out_cols if c in potential.columns]].copy()
    save_csv(out.sort_values("potential_capacity_rank_point"), root / "results/tables/potential_capacity_scores.csv")
    return out


def _model_dataset(root: Path, potential: pd.DataFrame) -> pd.DataFrame:
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix_expanded.csv")
    controls = matrix[
        [
            c
            for c in [
                "iso3",
                "gdp_per_capita_ppp",
                "age65_share",
                "population_density",
                "population_total",
                "expanded_feature_missing_rate",
            ]
            if c in matrix.columns
        ]
    ].drop_duplicates("iso3")
    data = potential.merge(outcomes[["iso3", "outcome_date", OUTCOME]], on="iso3", how="inner")
    data = data.merge(controls, on="iso3", how="left", suffixes=("", "_raw"))
    data["log_gdp_per_capita_ppp"] = np.log(data["gdp_per_capita_ppp"].where(data["gdp_per_capita_ppp"] > 0))
    data["population_density_log"] = np.log1p(data["population_density"].clip(lower=0))
    return data


def _preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    categorical = [c for c in feature_cols if df[c].dtype == "object" or str(df[c].dtype).startswith("category")]
    numeric = [c for c in feature_cols if c not in categorical]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def _pipeline(model_name: str, df: pd.DataFrame, feature_cols: list[str]) -> Pipeline:
    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "ridge":
        model = Ridge(alpha=10.0)
    elif model_name == "huber":
        model = HuberRegressor(max_iter=1000)
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=20260624)
    else:
        raise ValueError(model_name)
    return Pipeline([("prep", _preprocessor(df, feature_cols)), ("model", model)])


def _adjusted_r2(y: np.ndarray, pred: np.ndarray, p: int) -> float:
    if len(y) <= p + 1:
        return np.nan
    r2 = r2_score(y, pred)
    return float(1.0 - (1.0 - r2) * (len(y) - 1) / (len(y) - p - 1))


def _expected_burden_models(data: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [
        "potential_capacity_score",
        "log_gdp_per_capita_ppp",
        "age65_share",
        "population_density_log",
        "vulnerability_score",
        "region",
        "income_group",
    ]
    work = data[["iso3", "country", OUTCOME] + feature_cols].dropna().copy()
    models = ["linear", "ridge", "huber"]
    if len(work) >= 80:
        models.append("random_forest")
    rows = []
    prediction_rows = []
    cv = KFold(n_splits=min(5, max(2, len(work) // 20)), shuffle=True, random_state=20260624)
    x = work[feature_cols]
    y = work[OUTCOME].astype(float).to_numpy()
    for model_name in models:
        pipe = _pipeline(model_name, work, feature_cols)
        try:
            cv_pred = cross_val_predict(pipe, x, y, cv=cv)
            cv_rmse = float(mean_squared_error(y, cv_pred) ** 0.5)
        except Exception:
            cv_rmse = np.nan
        pipe.fit(x, y)
        pred = pipe.predict(x)
        transformed_p = pipe.named_steps["prep"].transform(x).shape[1]
        rows.append(
            {
                "model_version": model_name,
                "n": len(work),
                "n_candidate_validation_countries": data["iso3"].nunique(),
                "missing_count_due_to_model_covariates": int(data["iso3"].nunique() - len(work)),
                "missing_rate_due_to_model_covariates": float(1.0 - len(work) / max(data["iso3"].nunique(), 1)),
                "features": ",".join(feature_cols),
                "cv_rmse": cv_rmse,
                "in_sample_rmse": float(mean_squared_error(y, pred) ** 0.5),
                "r2": float(r2_score(y, pred)),
                "adjusted_r2": _adjusted_r2(y, pred, transformed_p),
                "status": "fit",
            }
        )
        for iso3, country, obs, predicted in zip(work["iso3"], work["country"], y, pred):
            prediction_rows.append(
                {
                    "iso3": iso3,
                    "country": country,
                    "model_version": model_name,
                    "observed_shock_burden": obs,
                    "predicted_shock_burden": float(predicted),
                    "residual_observed_minus_expected": float(obs - predicted),
                    "n_model_sample": len(work),
                    "n_candidate_validation_countries": data["iso3"].nunique(),
                    "missing_count_due_to_model_covariates": int(data["iso3"].nunique() - len(work)),
                    "missing_rate_due_to_model_covariates": float(1.0 - len(work) / max(data["iso3"].nunique(), 1)),
                    "cv_rmse": cv_rmse,
                }
            )
    expected = pd.DataFrame(prediction_rows)
    diagnostics = pd.DataFrame(rows)

    primary = expected[expected["model_version"].eq("ridge")].copy()
    if not primary.empty:
        intervals = _bootstrap_prediction_intervals(work, feature_cols, n_boot=250)
        expected = expected.merge(intervals, on=["iso3", "model_version"], how="left")
    save_csv(expected, root / "results/tables/expected_shock_burden.csv")
    save_csv(diagnostics, root / "results/tables/expected_shock_burden_model_diagnostics.csv")
    return expected, diagnostics


def _bootstrap_prediction_intervals(work: pd.DataFrame, feature_cols: list[str], n_boot: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(20260624)
    preds = []
    x_all = work[feature_cols]
    for _ in range(n_boot):
        idx = rng.integers(0, len(work), len(work))
        sample = work.iloc[idx]
        try:
            pipe = _pipeline("ridge", sample, feature_cols)
            pipe.fit(sample[feature_cols], sample[OUTCOME].astype(float))
            preds.append(pipe.predict(x_all))
        except Exception:
            continue
    if not preds:
        return pd.DataFrame({"iso3": work["iso3"], "model_version": "ridge"})
    arr = np.vstack(preds)
    return pd.DataFrame(
        {
            "iso3": work["iso3"].to_numpy(),
            "model_version": "ridge",
            "prediction_interval_low": np.nanpercentile(arr, 2.5, axis=0),
            "prediction_interval_high": np.nanpercentile(arr, 97.5, axis=0),
        }
    )


def _conversion_tables(root: Path, potential: pd.DataFrame, expected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = expected[expected["model_version"].eq("ridge")].copy()
    conv = primary.merge(potential, on=["iso3", "country"], how="left", suffixes=("", "_potential"))
    residual = conv["residual_observed_minus_expected"].astype(float)
    std = residual.std(ddof=0)
    conv["realized_resilience_gap"] = residual
    conv["gap_interpretation"] = np.where(
        conv["realized_resilience_gap"].gt(0),
        "worse-than-expected shock burden / capacity under-realization",
        "better-than-expected shock burden / adaptive over-performance",
    )
    conv["conversion_efficiency_zscore"] = -(residual - residual.mean()) / (std if std and np.isfinite(std) else 1.0)
    conv["expected_burden_percentile"] = conv["predicted_shock_burden"].rank(pct=True)
    conv["observed_burden_percentile"] = conv["observed_shock_burden"].rank(pct=True)
    conv["conversion_efficiency_percentile"] = conv["expected_burden_percentile"] - conv["observed_burden_percentile"]
    conv["higher_efficiency_is_better"] = True
    conv["n_validation_countries"] = len(conv)
    conv["missing_count_from_potential_capacity_sample"] = int(potential["iso3"].nunique() - len(conv))
    conv["missing_rate_from_potential_capacity_sample"] = float(1.0 - len(conv) / max(potential["iso3"].nunique(), 1))
    gap_cols = [
        "iso3",
        "country",
        "observed_shock_burden",
        "predicted_shock_burden",
        "realized_resilience_gap",
        "gap_interpretation",
        "prediction_interval_low",
        "prediction_interval_high",
        "n_validation_countries",
        "missing_count_from_potential_capacity_sample",
        "missing_rate_from_potential_capacity_sample",
    ]
    save_csv(conv[[c for c in gap_cols if c in conv.columns]], root / "results/tables/realized_resilience_gap.csv")
    eff_cols = [
        "iso3",
        "country",
        "potential_capacity_score",
        "median_potential_capacity_rank",
        "observed_shock_burden",
        "predicted_shock_burden",
        "realized_resilience_gap",
        "conversion_efficiency_zscore",
        "conversion_efficiency_percentile",
        "expected_burden_percentile",
        "observed_burden_percentile",
        "higher_efficiency_is_better",
        "method_disagreement_index",
        "expanded_feature_missing_rate",
        "n_validation_countries",
        "missing_count_from_potential_capacity_sample",
        "missing_rate_from_potential_capacity_sample",
    ]
    save_csv(conv[[c for c in eff_cols if c in conv.columns]].sort_values("conversion_efficiency_zscore", ascending=False), root / "results/tables/conversion_efficiency_scores.csv")
    return conv, conv[[c for c in eff_cols if c in conv.columns]]


def _assign_profiles(conv: pd.DataFrame, root: Path) -> pd.DataFrame:
    df = conv.copy()
    cap_median = df["potential_capacity_score"].median()
    burden_median = df["observed_shock_burden"].median()
    disagreement_cut = df["method_disagreement_index"].quantile(0.75)
    missing_cut = max(0.20, df["expanded_feature_missing_rate"].quantile(0.75))

    def label(row: pd.Series) -> str:
        if row["method_disagreement_index"] >= disagreement_cut or row["expanded_feature_missing_rate"] > missing_cut:
            return "uncertain / data-limited systems"
        high_capacity = row["potential_capacity_score"] >= cap_median
        high_burden = row["observed_shock_burden"] > burden_median
        positive_eff = row["conversion_efficiency_zscore"] >= 0
        if high_capacity and not high_burden and positive_eff:
            return "effective converters"
        if high_capacity and (high_burden or not positive_eff):
            return "capacity under-realizers"
        if (not high_capacity) and positive_eff:
            return "adaptive over-performers"
        return "structurally vulnerable systems"

    df["conversion_profile"] = df.apply(label, axis=1)
    df["profile_rule"] = (
        "uncertain if top-quartile method disagreement or high missingness; otherwise median split on potential capacity, observed burden, and conversion efficiency"
    )
    out_cols = [
        "iso3",
        "country",
        "region",
        "income_group",
        "conversion_profile",
        "profile_rule",
        "potential_capacity_score",
        "observed_shock_burden",
        "predicted_shock_burden",
        "realized_resilience_gap",
        "conversion_efficiency_zscore",
        "conversion_efficiency_percentile",
        "capacity_score",
        "preparedness_score",
        "vulnerability_score",
        "equity_access_score",
        "method_disagreement_index",
        "expanded_feature_missing_rate",
        "n_validation_countries",
    ]
    profiles = df[[c for c in out_cols if c in df.columns]].copy()
    profiles["profile_n"] = profiles.groupby("conversion_profile")["iso3"].transform("count")
    profiles["profile_missing_rate_mean"] = profiles.groupby("conversion_profile")["expanded_feature_missing_rate"].transform("mean")
    save_csv(profiles.sort_values(["conversion_profile", "conversion_efficiency_zscore"], ascending=[True, False]), root / "results/tables/conversion_profiles.csv")
    return profiles


def _case_studies(profiles: pd.DataFrame, root: Path) -> pd.DataFrame:
    selections = []
    rules = [
        ("capacity under-realizers", True),
        ("adaptive over-performers", False),
        ("effective converters", False),
        ("structurally vulnerable systems", True),
        ("uncertain / data-limited systems", True),
    ]
    for profile, ascending in rules:
        sub = profiles[profiles["conversion_profile"].eq(profile)].copy()
        if sub.empty:
            continue
        if profile == "uncertain / data-limited systems":
            sub = sub.sort_values("method_disagreement_index", ascending=False)
        else:
            sub = sub.sort_values("conversion_efficiency_zscore", ascending=ascending)
        selections.append(sub.head(2))
    cases = pd.concat(selections, ignore_index=True).drop_duplicates("iso3").head(12)
    block_cols = ["capacity_score", "preparedness_score", "vulnerability_score", "equity_access_score"]
    cases["likely_conversion_bottleneck_blocks"] = cases[block_cols].apply(
        lambda row: ",".join(row.sort_values().head(2).index.str.replace("_score", "")),
        axis=1,
    )
    cases["diagnostic_note"] = "Decision-support scenario only; no causal policy claim."
    save_csv(cases, root / "results/tables/conversion_case_studies.csv")
    return cases


def _plot_outputs(conv: pd.DataFrame, profiles: pd.DataFrame, root: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(conv["potential_capacity_score"], conv["observed_shock_burden"], c=conv["conversion_efficiency_zscore"], cmap="coolwarm_r", s=35, alpha=0.85)
    plt.colorbar(label="Conversion efficiency z-score")
    plt.xlabel("Potential capacity score")
    plt.ylabel("Observed shock burden")
    plt.title("Potential capacity versus shock burden")
    plt.tight_layout()
    plt.savefig(root / "results/figures/potential_capacity_vs_shock_burden.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    for profile, sub in profiles.groupby("conversion_profile"):
        plt.scatter(sub["potential_capacity_score"], sub["conversion_efficiency_zscore"], label=profile, s=30, alpha=0.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(profiles["potential_capacity_score"].median(), color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Potential capacity score")
    plt.ylabel("Conversion efficiency z-score")
    plt.title("Conversion efficiency quadrants")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(root / "results/figures/conversion_efficiency_quadrants.pdf", bbox_inches="tight")
    plt.close()

    rank_df = conv.copy()
    rank_df["capacity_rank"] = rank_df["potential_capacity_score"].rank(ascending=False, method="min")
    rank_df["efficiency_rank"] = rank_df["conversion_efficiency_zscore"].rank(ascending=False, method="min")
    rank_df["rank_shift_efficiency_minus_capacity"] = rank_df["efficiency_rank"] - rank_df["capacity_rank"]
    plot = pd.concat([rank_df.nsmallest(15, "rank_shift_efficiency_minus_capacity"), rank_df.nlargest(15, "rank_shift_efficiency_minus_capacity")])
    plt.figure(figsize=(9, 7))
    plt.barh(plot["country"], plot["rank_shift_efficiency_minus_capacity"], color=np.where(plot["rank_shift_efficiency_minus_capacity"] < 0, "#31a354", "#de2d26"))
    plt.axvline(0, color="black", linewidth=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Efficiency rank minus capacity rank")
    plt.title("Conversion efficiency rank shift")
    plt.tight_layout()
    plt.savefig(root / "results/figures/conversion_efficiency_rank_shift.pdf", bbox_inches="tight")
    plt.close()

    extremes = pd.concat([conv.nsmallest(10, "conversion_efficiency_zscore"), conv.nlargest(10, "conversion_efficiency_zscore")]).sort_values("conversion_efficiency_zscore")
    plt.figure(figsize=(9, 7))
    plt.barh(extremes["country"], extremes["conversion_efficiency_zscore"], color=np.where(extremes["conversion_efficiency_zscore"] >= 0, "#31a354", "#de2d26"))
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Conversion efficiency z-score")
    plt.title("Top under-realizers and over-performers")
    plt.tight_layout()
    plt.savefig(root / "results/figures/top_under_realizers_and_over_performers.pdf", bbox_inches="tight")
    plt.close()


def run_resilience_conversion(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    potential = _potential_capacity_table(root)
    data = _model_dataset(root, potential)
    expected, _diagnostics = _expected_burden_models(data, root)
    conv, efficiency = _conversion_tables(root, potential, expected)
    profiles = _assign_profiles(conv, root)
    _case_studies(profiles, root)
    _plot_outputs(conv, profiles, root)
    return efficiency, profiles


if __name__ == "__main__":
    run_resilience_conversion()
