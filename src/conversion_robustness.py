from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import ROOT, ensure_dirs, save_csv


OUTCOME = "cumulative_excess_deaths_per_million_2020_2022"
PROFILE_LABELS = [
    "effective converters",
    "capacity under-realizers",
    "adaptive over-performers",
    "structurally vulnerable systems",
    "uncertain / data-limited systems",
]


def load_conversion_model_data(root: Path = ROOT, outcome_col: str = OUTCOME) -> pd.DataFrame:
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
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
            ]
            if c in matrix.columns
        ]
    ].drop_duplicates("iso3")
    data = potential.merge(outcomes[["iso3", OUTCOME]], on="iso3", how="inner")
    if outcome_col != OUTCOME and outcome_col in outcomes.columns:
        data[outcome_col] = outcomes[outcome_col]
    data = data.merge(controls, on="iso3", how="left")
    data["log_gdp_per_capita_ppp"] = np.log(data["gdp_per_capita_ppp"].where(data["gdp_per_capita_ppp"] > 0))
    data["population_density_log"] = np.log1p(data["population_density"].clip(lower=0))
    return data


def model_feature_columns() -> list[str]:
    return [
        "potential_capacity_score",
        "log_gdp_per_capita_ppp",
        "age65_share",
        "population_density_log",
        "vulnerability_score",
        "region",
        "income_group",
    ]


def encoded_xy(data: pd.DataFrame, outcome_col: str = OUTCOME, feature_cols: list[str] | None = None) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    feature_cols = feature_cols or model_feature_columns()
    extra_cols = [
        c
        for c in ["method_disagreement_index", "expanded_feature_missing_rate", "population_total"]
        if c in data.columns and c not in feature_cols
    ]
    work = data[["iso3", "country", outcome_col] + feature_cols + extra_cols].dropna(subset=[outcome_col] + feature_cols).copy()
    x = pd.get_dummies(work[feature_cols], columns=[c for c in feature_cols if work[c].dtype == "object"], drop_first=True)
    x = x.astype(float)
    y = work[outcome_col].astype(float)
    return x, y, work


def numeric_scaler(x: pd.DataFrame) -> ColumnTransformer:
    """Standardize numeric columns only and pass one-hot dummies through unchanged.

    This mirrors the Stage 2 preprocessor in ``resilience_conversion._preprocessor`` so that every
    ridge refit (bootstrap, Monte Carlo, GHS-enhanced and life-expectancy variants) uses exactly the
    same model as the baseline conversion table.
    """

    numeric = [c for c in x.columns if not set(pd.unique(x[c].dropna())).issubset({0.0, 1.0})]
    return ColumnTransformer([("num", StandardScaler(), numeric)], remainder="passthrough")


def _model(model_name: str, x: pd.DataFrame | None = None):
    scaler = numeric_scaler(x) if x is not None else StandardScaler()
    if model_name == "linear":
        return Pipeline([("scale", scaler), ("model", LinearRegression())])
    if model_name == "ridge":
        return Pipeline([("scale", scaler), ("model", Ridge(alpha=10.0))])
    if model_name == "huber":
        return Pipeline([("scale", scaler), ("model", HuberRegressor(max_iter=1000))])
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=20260624)
    raise ValueError(model_name)


def _efficiency_from_prediction(work: pd.DataFrame, predicted: np.ndarray, model_name: str) -> pd.DataFrame:
    out = work[[
        "iso3",
        "country",
        "region",
        "income_group",
        "potential_capacity_score",
        "observed_shock_burden" if "observed_shock_burden" in work.columns else OUTCOME,
        "method_disagreement_index",
        "expanded_feature_missing_rate",
    ]].copy()
    if OUTCOME in out.columns:
        out = out.rename(columns={OUTCOME: "observed_shock_burden"})
    out["model_version"] = model_name
    out["predicted_shock_burden"] = predicted
    out["realized_resilience_gap"] = out["observed_shock_burden"] - out["predicted_shock_burden"]
    std = out["realized_resilience_gap"].std(ddof=0)
    out["conversion_efficiency_zscore"] = -(
        out["realized_resilience_gap"] - out["realized_resilience_gap"].mean()
    ) / (std if std and np.isfinite(std) else 1.0)
    out["conversion_profile"] = assign_conversion_profiles(out)
    return out


def assign_conversion_profiles(df: pd.DataFrame) -> pd.Series:
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

    return df.apply(label, axis=1)


def fit_conversion_models(root: Path = ROOT) -> pd.DataFrame:
    data = load_conversion_model_data(root)
    x, y, work = encoded_xy(data)
    work = work.rename(columns={OUTCOME: "observed_shock_burden"})
    rows = []
    for model_name in ["linear", "ridge", "huber", "random_forest"]:
        model = _model(model_name, x)
        model.fit(x, y)
        pred = model.predict(x)
        rows.append(_efficiency_from_prediction(work, pred, model_name))
    return pd.concat(rows, ignore_index=True)


def run_model_agreement(root: Path = ROOT) -> pd.DataFrame:
    predictions = fit_conversion_models(root)
    rows = []
    for a, b in combinations(sorted(predictions["model_version"].unique()), 2):
        left = predictions[predictions["model_version"].eq(a)]
        right = predictions[predictions["model_version"].eq(b)]
        merged = left.merge(
            right,
            on="iso3",
            suffixes=("_a", "_b"),
            how="inner",
        )
        n = len(merged)
        under_a = set(merged.nsmallest(10, "conversion_efficiency_zscore_a")["iso3"])
        under_b = set(merged.nsmallest(10, "conversion_efficiency_zscore_b")["iso3"])
        over_a = set(merged.nlargest(10, "conversion_efficiency_zscore_a")["iso3"])
        over_b = set(merged.nlargest(10, "conversion_efficiency_zscore_b")["iso3"])
        rows.append(
            {
                "model_a": a,
                "model_b": b,
                "n": n,
                "coverage_count": n,
                "missing_count": int(predictions["iso3"].nunique() - n),
                "missing_rate": float(1.0 - n / max(predictions["iso3"].nunique(), 1)),
                "spearman_efficiency": spearmanr(
                    merged["conversion_efficiency_zscore_a"], merged["conversion_efficiency_zscore_b"]
                ).statistic,
                "kendall_efficiency": kendalltau(
                    merged["conversion_efficiency_zscore_a"], merged["conversion_efficiency_zscore_b"]
                ).statistic,
                "top10_under_realizer_overlap": len(under_a & under_b) / 10,
                "top10_over_performer_overlap": len(over_a & over_b) / 10,
                "profile_assignment_agreement": float(
                    (merged["conversion_profile_a"] == merged["conversion_profile_b"]).mean()
                ),
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/conversion_model_agreement.csv")
    _plot_model_agreement(out, root)
    return out


def _plot_model_agreement(out: pd.DataFrame, root: Path) -> None:
    models = sorted(set(out["model_a"]) | set(out["model_b"]))
    mat = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
    for _, row in out.iterrows():
        mat.loc[row["model_a"], row["model_b"]] = row["spearman_efficiency"]
        mat.loc[row["model_b"], row["model_a"]] = row["spearman_efficiency"]
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(models)), models, rotation=45, ha="right")
    plt.yticks(range(len(models)), models)
    plt.title("Conversion efficiency model agreement")
    plt.colorbar(im, label="Spearman correlation")
    plt.tight_layout()
    plt.savefig(root / "results/figures/model_agreement_heatmap.pdf", bbox_inches="tight")
    plt.close()


def bootstrap_conversion_uncertainty(root: Path = ROOT, n_boot: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_conversion_model_data(root)
    x, y, work = encoded_xy(data)
    work = work.rename(columns={OUTCOME: "observed_shock_burden"}).reset_index(drop=True)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    rng = np.random.default_rng(20260624)
    eff = np.full((n_boot, len(work)), np.nan)
    profile_records: list[list[str]] = []
    for b in range(n_boot):
        idx = rng.integers(0, len(work), len(work))
        model = _model("ridge", x)
        model.fit(x.iloc[idx], y.iloc[idx])
        pred = model.predict(x)
        pred_df = _efficiency_from_prediction(work, pred, "ridge_bootstrap")
        eff[b, :] = pred_df["conversion_efficiency_zscore"].to_numpy()
        profile_records.append(pred_df["conversion_profile"].tolist())

    profiles = pd.DataFrame(profile_records, columns=work["iso3"])
    rows = []
    profile_rows = []
    for j, (_, row) in enumerate(work.iterrows()):
        vals = eff[:, j]
        probs = profiles[row["iso3"]].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "iso3": row["iso3"],
                "country": row["country"],
                "n_bootstrap": n_boot,
                "coverage_count": len(work),
                "missing_count_from_potential_capacity_sample": int(
                    pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique() - len(work)
                ),
                "missing_rate_from_potential_capacity_sample": float(
                    1.0
                    - len(work)
                    / max(pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique(), 1)
                ),
                "conversion_efficiency_mean": float(np.nanmean(vals)),
                "conversion_efficiency_std": float(np.nanstd(vals)),
                "conversion_efficiency_ci_low": float(np.nanpercentile(vals, 2.5)),
                "conversion_efficiency_ci_high": float(np.nanpercentile(vals, 97.5)),
                "probability_under_realizer": float(probs.get("capacity under-realizers", 0.0)),
                "probability_over_performer": float(probs.get("adaptive over-performers", 0.0)),
                "probability_data_limited_uncertain": float(probs.get("uncertain / data-limited systems", 0.0)),
                "profile_stability_score": float(max(probs.values()) if probs else np.nan),
            }
        )
        for label in PROFILE_LABELS:
            profile_rows.append(
                {
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "profile": label,
                    "profile_probability": float(probs.get(label, 0.0)),
                    "n_bootstrap": n_boot,
                    "coverage_count": len(work),
                    "missing_count": 0,
                    "missing_rate": 0.0,
                }
            )
    summary = pd.DataFrame(rows)
    profile_stability = pd.DataFrame(profile_rows)
    save_csv(summary, root / "results/tables/conversion_efficiency_uncertainty.csv")
    save_csv(profile_stability, root / "results/tables/conversion_profile_stability.csv")
    _plot_efficiency_intervals(summary, root / "results/figures/conversion_efficiency_interval_plot.pdf")
    _plot_profile_stability(summary, root / "results/figures/profile_stability_barplot.pdf")
    return summary, profile_stability


def _plot_efficiency_intervals(summary: pd.DataFrame, path: Path) -> None:
    plot = pd.concat([summary.nsmallest(20, "conversion_efficiency_mean"), summary.nlargest(20, "conversion_efficiency_mean")])
    plot = plot.sort_values("conversion_efficiency_mean")
    y = np.arange(len(plot))
    plt.figure(figsize=(9, max(7, 0.18 * len(plot))))
    plt.hlines(y, plot["conversion_efficiency_ci_low"], plot["conversion_efficiency_ci_high"], color="#9ecae1")
    plt.scatter(plot["conversion_efficiency_mean"], y, color="#08519c", s=14)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.yticks(y, plot["country"], fontsize=7)
    plt.xlabel("Bootstrap conversion efficiency")
    plt.title("Conversion efficiency uncertainty intervals")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_profile_stability(summary: pd.DataFrame, path: Path) -> None:
    plot = summary.sort_values("profile_stability_score").head(40)
    plt.figure(figsize=(9, 7))
    plt.barh(plot["country"], plot["profile_stability_score"], color="#756bb1")
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.xlabel("Max bootstrap profile probability")
    plt.title("Least stable conversion profiles")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def leave_group_out_stability(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(root / "results/tables/conversion_profiles.csv")[
        ["iso3", "country", "region", "income_group", "conversion_profile", "conversion_efficiency_zscore"]
    ].rename(
        columns={
            "conversion_profile": "baseline_profile",
            "conversion_efficiency_zscore": "baseline_conversion_efficiency",
        }
    )
    data = load_conversion_model_data(root)
    x, y, work = encoded_xy(data)
    work = work.rename(columns={OUTCOME: "observed_shock_burden"}).reset_index(drop=True)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)

    scenarios: list[tuple[str, pd.Series]] = []
    for region in sorted(work["region"].dropna().unique()):
        scenarios.append((f"leave_region_out::{region}", ~work["region"].eq(region)))
    for income in sorted(work["income_group"].dropna().unique()):
        scenarios.append((f"leave_income_group_out::{income}", ~work["income_group"].eq(income)))
    scenarios.append(("exclude_high_income", ~work["income_group"].eq("High income")))
    scenarios.append(("exclude_non_high_income", work["income_group"].eq("High income")))
    if "population_total" in data.columns:
        pop = data.set_index("iso3").loc[work["iso3"], "population_total"].reset_index(drop=True)
        scenarios.append(("exclude_population_below_1m", pop.ge(1_000_000)))
    lo, hi = work["observed_shock_burden"].quantile([0.025, 0.975])
    scenarios.append(("exclude_top_bottom_2_5pct_outcomes", work["observed_shock_burden"].between(lo, hi)))

    rows = []
    outlier_rows = []
    for name, mask in scenarios:
        if mask.sum() < 30:
            continue
        model = _model("ridge")
        model.fit(x.loc[mask], y.loc[mask])
        pred_work = work.loc[mask].reset_index(drop=True)
        pred = model.predict(x.loc[mask])
        scenario = _efficiency_from_prediction(pred_work, pred, "ridge_subset")
        merged = scenario.merge(base, on=["iso3", "country"], how="left")
        merged["scenario"] = name
        merged["n"] = len(merged)
        merged["coverage_count"] = len(merged)
        merged["missing_count_from_full_validation"] = int(len(work) - len(merged))
        merged["missing_rate_from_full_validation"] = float(1.0 - len(merged) / max(len(work), 1))
        merged["efficiency_delta_vs_baseline"] = (
            merged["conversion_efficiency_zscore"] - merged["baseline_conversion_efficiency"]
        )
        merged["profile_stable"] = merged["conversion_profile"] == merged["baseline_profile"]
        keep = [
            "scenario",
            "iso3",
            "country",
            "n",
            "coverage_count",
            "missing_count_from_full_validation",
            "missing_rate_from_full_validation",
            "baseline_conversion_efficiency",
            "conversion_efficiency_zscore",
            "efficiency_delta_vs_baseline",
            "baseline_profile",
            "conversion_profile",
            "profile_stable",
        ]
        if name.startswith("leave_"):
            rows.append(merged[keep])
        else:
            outlier_rows.append(merged[keep])
    leave = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    outlier = pd.concat(outlier_rows, ignore_index=True) if outlier_rows else pd.DataFrame()
    save_csv(leave, root / "results/tables/leave_group_out_conversion_stability.csv")
    save_csv(outlier, root / "results/tables/outlier_conversion_sensitivity.csv")
    return leave, outlier


def outcome_window_profile_stability(root: Path = ROOT) -> pd.DataFrame:
    # Uses outcome windows already audited in Stage 2 and stores overlap rows in the profile stability table.
    windows = pd.read_csv(root / "results/tables/outcome_window_sensitivity.csv")["outcome_window"].unique()
    base_profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    rows = []
    for window in windows:
        rows.append(
            {
                "stability_source": "outcome_window",
                "comparison": f"{window}_vs_2020_2022",
                "n": len(base_profiles),
                "coverage_count": len(base_profiles),
                "missing_count": 0,
                "missing_rate": 0.0,
                "profile_overlap_rate": 1.0 if window in {"2022_end", "2020_2022"} else np.nan,
                "note": "Outcome-window conversion recomputation is approximated by the Stage 2 outcome-window score audit unless an alternate outcome file is added.",
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/outcome_window_conversion_profile_overlap.csv")
    return out


def run_conversion_robustness(root: Path = ROOT, n_boot: int = 1000) -> None:
    ensure_dirs(root)
    run_model_agreement(root)
    bootstrap_conversion_uncertainty(root, n_boot=n_boot)
    leave_group_out_stability(root)
    outcome_window_profile_stability(root)


if __name__ == "__main__":
    run_conversion_robustness()
