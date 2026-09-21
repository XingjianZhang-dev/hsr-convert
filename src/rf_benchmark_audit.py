from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .conversion_robustness import load_conversion_model_data
from .utils import ROOT, ensure_dirs, save_csv


OUTCOME = "cumulative_excess_deaths_per_million_2020_2022"
RF_FEATURES = ["capacity_score", "preparedness_score", "vulnerability_score", "equity_access_score"]
FORBIDDEN_TERMS = ["outcome", "shock", "excess", "death", "covid", "case", "testing", "vaccination", "stringency"]


def _dataset(root: Path) -> pd.DataFrame:
    data = load_conversion_model_data(root)
    return data[["iso3", "country", "region", "income_group", OUTCOME] + RF_FEATURES].dropna().copy()


def _model(name: str):
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=20260624)
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
    if name == "linear":
        return Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    raise ValueError(name)


def _cv_predictions(x: pd.DataFrame, y: pd.Series, model_name: str, repeats: int = 10) -> tuple[np.ndarray, list[float]]:
    all_pred = np.zeros((repeats, len(y)))
    rmses = []
    for rep in range(repeats):
        cv = KFold(n_splits=5, shuffle=True, random_state=20260624 + rep)
        pred = np.full(len(y), np.nan)
        for train, test in cv.split(x):
            model = _model(model_name)
            model.fit(x.iloc[train], y.iloc[train])
            pred[test] = model.predict(x.iloc[test])
        all_pred[rep, :] = pred
        rmses.append(float(mean_squared_error(y, pred) ** 0.5))
    return all_pred.mean(axis=0), rmses


def run_rf_benchmark_audit(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    data = _dataset(root)
    x = data[RF_FEATURES].astype(float)
    y = data[OUTCOME].astype(float)
    leakage_terms_found = [f for f in RF_FEATURES if any(term in f.lower() for term in FORBIDDEN_TERMS)]
    rows = []
    cv_rows = []
    predictions: dict[str, np.ndarray] = {}
    for model_name in ["random_forest", "ridge", "linear"]:
        pred, rmses = _cv_predictions(x, y, model_name)
        predictions[model_name] = pred
        rows.append(
            {
                "audit_item": f"{model_name}_repeated_kfold_cv",
                "model": model_name,
                "features": ",".join(RF_FEATURES),
                "n": len(data),
                "coverage_count": len(data),
                "missing_count": int(pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique() - len(data)),
                "missing_rate": float(
                    1.0 - len(data) / max(pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique(), 1)
                ),
                "leakage_terms_found": ",".join(leakage_terms_found),
                "no_leakage_detected": len(leakage_terms_found) == 0,
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "spearman_predicted_observed": spearmanr(pred, y).statistic,
                "r2_cv": r2_score(y, pred),
                "status": "sensitivity_only" if model_name == "random_forest" else "benchmark",
            }
        )
        for i, rmse in enumerate(rmses):
            cv_rows.append({"model": model_name, "cv_scheme": "repeated_kfold", "repeat": i, "rmse": rmse, "n": len(data)})

    for group_col, scheme in [("region", "leave_one_region_out"), ("income_group", "leave_one_income_group_out")]:
        for group in sorted(data[group_col].dropna().unique()):
            train = ~data[group_col].eq(group)
            test = data[group_col].eq(group)
            if train.sum() < 30 or test.sum() < 3:
                continue
            for model_name in ["random_forest", "ridge", "linear"]:
                model = _model(model_name)
                model.fit(x.loc[train], y.loc[train])
                pred = model.predict(x.loc[test])
                cv_rows.append(
                    {
                        "model": model_name,
                        "cv_scheme": scheme,
                        "held_out_group": group,
                        "rmse": float(mean_squared_error(y.loc[test], pred) ** 0.5),
                        "spearman_predicted_observed": spearmanr(pred, y.loc[test]).statistic if test.sum() >= 4 else np.nan,
                        "n": int(test.sum()),
                    }
                )

    rng = np.random.default_rng(20260624)
    shuffled = y.sample(frac=1.0, random_state=20260624).reset_index(drop=True)
    shuffled_pred, shuffled_rmses = _cv_predictions(x.reset_index(drop=True), shuffled, "random_forest", repeats=10)
    rows.append(
        {
            "audit_item": "random_forest_shuffled_outcome_negative_control",
            "model": "random_forest",
            "features": ",".join(RF_FEATURES),
            "n": len(data),
            "coverage_count": len(data),
            "missing_count": int(pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique() - len(data)),
            "missing_rate": float(
                1.0 - len(data) / max(pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique(), 1)
            ),
            "leakage_terms_found": ",".join(leakage_terms_found),
            "no_leakage_detected": len(leakage_terms_found) == 0,
            "rmse_mean": float(np.mean(shuffled_rmses)),
            "rmse_std": float(np.std(shuffled_rmses)),
            "spearman_predicted_observed": spearmanr(shuffled_pred, shuffled).statistic,
            "r2_cv": r2_score(shuffled, shuffled_pred),
            "status": "negative_control",
        }
    )

    rf = _model("random_forest")
    rf.fit(x, y)
    importances = permutation_importance(rf, x, y, n_repeats=50, random_state=20260624)
    importance = pd.DataFrame(
        {
            "feature": RF_FEATURES,
            "importance_mean": importances.importances_mean,
            "importance_std": importances.importances_std,
            "n": len(data),
            "coverage_count": len(data),
            "missing_count": 0,
            "missing_rate": 0.0,
            "leakage_screen_passed": len(leakage_terms_found) == 0,
        }
    ).sort_values("importance_mean", ascending=False)
    audit = pd.DataFrame(rows)
    cv = pd.DataFrame(cv_rows)
    save_csv(audit, root / "results/tables/rf_benchmark_audit.csv")
    save_csv(importance, root / "results/tables/rf_permutation_importance.csv")
    save_csv(cv, root / "results/tables/rf_cv_sensitivity.csv")
    _plot_predicted_observed(data, predictions["random_forest"], root)
    _plot_importance(importance, root)
    return audit, importance


def _plot_predicted_observed(data: pd.DataFrame, pred: np.ndarray, root: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(data[OUTCOME], pred, s=28, alpha=0.75)
    lo = min(data[OUTCOME].min(), np.nanmin(pred))
    hi = max(data[OUTCOME].max(), np.nanmax(pred))
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    plt.xlabel("Observed shock burden")
    plt.ylabel("RF cross-validated predicted burden")
    plt.title("RF predicted versus observed, repeated K-fold")
    plt.tight_layout()
    plt.savefig(root / "results/figures/rf_predicted_vs_observed.pdf", bbox_inches="tight")
    plt.close()


def _plot_importance(importance: pd.DataFrame, root: Path) -> None:
    plot = importance.sort_values("importance_mean")
    plt.figure(figsize=(7, 4))
    plt.barh(plot["feature"], plot["importance_mean"], xerr=plot["importance_std"], color="#3182bd")
    plt.xlabel("Permutation importance")
    plt.title("RF permutation importance")
    plt.tight_layout()
    plt.savefig(root / "results/figures/rf_permutation_importance.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_rf_benchmark_audit()
