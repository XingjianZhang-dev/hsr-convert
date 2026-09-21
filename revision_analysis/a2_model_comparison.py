"""A2: expanded expected-burden model comparison under one repeated K-fold protocol [R2-5]."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedKFold

from revision_analysis.common import RESULTS_DIR, SEED, ensure_results_dirs, make_model, model_data, update_numbers

MODELS = ["linear", "ridge", "huber", "elastic_net", "random_forest", "gradient_boosting"]
N_SPLITS = 5
N_REPEATS = 20


def corrected_resampled_ttest(diff: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated-CV paired differences.

    The correction inflates the variance of the mean difference by (1/k + n_test/n_train) instead of 1/k,
    because training sets overlap across folds and repeats, which makes the naive paired t-test anti-conservative.
    """

    k = len(diff)
    mean = diff.mean()
    var = diff.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    se = np.sqrt(var * (1.0 / k + n_test / n_train))
    t = mean / se
    p = 2.0 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def run_repeated_cv() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    x, y, work = model_data()
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    folds = list(rkf.split(x))
    fold_rows = []
    oof = {m: np.full((N_REPEATS, len(y)), np.nan) for m in MODELS + ["intercept_only"]}
    for name in MODELS + ["intercept_only"]:
        for f, (tr, te) in enumerate(folds):
            repeat = f // N_SPLITS
            model = make_model(name, x)
            model.fit(x.iloc[tr], y.iloc[tr])
            pred = model.predict(x.iloc[te])
            oof[name][repeat, te] = pred
            err = y.iloc[te].to_numpy() - pred
            fold_rows.append(
                {
                    "model": name,
                    "repeat": repeat,
                    "fold": f % N_SPLITS,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "rmse": float(np.sqrt(np.mean(err**2))),
                    "mae": float(np.mean(np.abs(err))),
                }
            )
    fold_df = pd.DataFrame(fold_rows)
    # R^2 per repeat from pooled out-of-fold predictions (each country predicted exactly once per repeat).
    r2_rows = []
    y_arr = y.to_numpy()
    for name in MODELS + ["intercept_only"]:
        for r in range(N_REPEATS):
            pred = oof[name][r]
            ss_res = float(np.sum((y_arr - pred) ** 2))
            ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
            r2_rows.append({"model": name, "repeat": r, "cv_r2": 1.0 - ss_res / ss_tot, "cv_rmse_pooled": float(np.sqrt(ss_res / len(y_arr)))})
    r2_df = pd.DataFrame(r2_rows)
    return fold_df, r2_df, oof


def summarize(fold_df: pd.DataFrame, r2_df: pd.DataFrame) -> pd.DataFrame:
    n_train = int(fold_df["n_train"].iloc[0])
    n_test = int(round(fold_df["n_test"].mean()))
    ridge = fold_df[fold_df.model.eq("ridge")].sort_values(["repeat", "fold"])["rmse"].to_numpy()
    ridge_rep = r2_df[r2_df.model.eq("ridge")].sort_values("repeat")["cv_rmse_pooled"].to_numpy()
    rows = []
    for name in MODELS + ["intercept_only"]:
        sub = fold_df[fold_df.model.eq(name)].sort_values(["repeat", "fold"])
        rep = r2_df[r2_df.model.eq(name)].sort_values("repeat")
        row = {
            "model": name,
            "n_countries": int(sub["n_train"].iloc[0] + sub["n_test"].iloc[0]),
            "cv_protocol": f"{N_SPLITS}-fold x {N_REPEATS} repeats, seed {SEED}, identical folds for all models",
            "cv_rmse_mean": float(sub["rmse"].mean()),
            "cv_rmse_sd": float(sub["rmse"].std(ddof=1)),
            "cv_mae_mean": float(sub["mae"].mean()),
            "cv_mae_sd": float(sub["mae"].std(ddof=1)),
            "cv_r2_mean": float(rep["cv_r2"].mean()),
            "cv_r2_sd": float(rep["cv_r2"].std(ddof=1)),
        }
        if name != "ridge":
            diff = sub["rmse"].to_numpy() - ridge  # positive = worse than ridge
            t, p = corrected_resampled_ttest(diff, n_train, n_test)
            w = stats.wilcoxon(rep["cv_rmse_pooled"].to_numpy() - ridge_rep, zero_method="wilcox", alternative="two-sided")
            row.update(
                {
                    "delta_rmse_vs_ridge_mean": float(diff.mean()),
                    "delta_rmse_vs_ridge_sd": float(diff.std(ddof=1)),
                    "corrected_t_vs_ridge": t,
                    "corrected_t_p_vs_ridge": p,
                    "wilcoxon_p_vs_ridge_per_repeat": float(w.pvalue),
                }
            )
        else:
            row.update(
                {
                    "delta_rmse_vs_ridge_mean": 0.0,
                    "delta_rmse_vs_ridge_sd": 0.0,
                    "corrected_t_vs_ridge": np.nan,
                    "corrected_t_p_vs_ridge": np.nan,
                    "wilcoxon_p_vs_ridge_per_repeat": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    ensure_results_dirs()
    fold_df, r2_df, oof = run_repeated_cv()
    summary = summarize(fold_df, r2_df)
    fold_df.to_csv(RESULTS_DIR / "model_comparison_folds.csv", index=False)
    r2_df.to_csv(RESULTS_DIR / "model_comparison_repeats.csv", index=False)
    summary.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    # Out-of-fold predictions (averaged over repeats) for the calibration analysis in A4.
    x, y, work = model_data()
    oof_df = work[["iso3", "country", "observed_shock_burden"]].copy()
    for name in MODELS:
        oof_df[f"oof_pred_{name}"] = oof[name].mean(axis=0)
    oof_df.to_csv(RESULTS_DIR / "model_comparison_oof_predictions.csv", index=False)
    np.save(RESULTS_DIR / "model_comparison_oof_ridge_by_repeat.npy", oof["ridge"])
    payload = {"protocol": summary["cv_protocol"].iloc[0], "n_splits": N_SPLITS, "n_repeats": N_REPEATS, "seed": SEED}
    for _, r in summary.iterrows():
        payload[r["model"]] = {k: r[k] for k in summary.columns if k not in {"model", "cv_protocol"}}
    best = summary[summary.model.ne("intercept_only")].sort_values("cv_rmse_mean").iloc[0]
    payload["best_model_by_cv_rmse"] = best["model"]
    payload["ridge_rank_by_cv_rmse"] = int(summary[summary.model.ne("intercept_only")].sort_values("cv_rmse_mean").reset_index().query("model=='ridge'").index[0] + 1)
    update_numbers("model_comparison", payload)
    return summary


if __name__ == "__main__":
    print(run().round(3).to_string())
