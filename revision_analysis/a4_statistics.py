"""A4: permutation tests, bootstrap CIs, calibration and profile-difference checks [R2-7]."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from statistics import NormalDist

from revision_analysis.common import ROOT, RESULTS_DIR, SEED, ensure_results_dirs, make_model, model_data, update_numbers

N_PERM = 10000
N_BOOT = 2000


# ---------------------------------------------------------------------------------------------
# 1. permutation tests
# ---------------------------------------------------------------------------------------------
def _cv_rmse(model_name: str, x: pd.DataFrame, y: np.ndarray, folds: list) -> float:
    err = np.empty(len(y))
    for tr, te in folds:
        m = make_model(model_name, x).fit(x.iloc[tr], y[tr])
        err[te] = y[te] - m.predict(x.iloc[te])
    return float(np.sqrt(np.mean(err**2)))


def permutation_tests() -> dict:
    x, y, work = model_data()
    y_arr = y.to_numpy()
    rng = np.random.default_rng(SEED)
    # (a) capacity vs observed burden, Spearman, permuting the outcome
    c = work["potential_capacity_score"].to_numpy()
    obs = stats.spearmanr(c, y_arr).statistic
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = stats.spearmanr(c, rng.permutation(y_arr)).statistic
    p_cap = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))
    # (b) incremental predictive value of the full ridge feature set over an intercept-only model:
    # statistic = CV RMSE(intercept) - CV RMSE(ridge) on a fixed 5-fold split; outcome permuted under H0.
    folds = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(x))
    rmse_int = _cv_rmse("intercept_only", x, y_arr, folds)
    rmse_ridge = _cv_rmse("ridge", x, y_arr, folds)
    obs_gain = rmse_int - rmse_ridge
    null_gain = np.empty(N_PERM)
    for i in range(N_PERM):
        yp = rng.permutation(y_arr)
        null_gain[i] = _cv_rmse("intercept_only", x, yp, folds) - _cv_rmse("ridge", x, yp, folds)
    p_gain = float((np.sum(null_gain >= obs_gain) + 1) / (N_PERM + 1))
    # (c) the same statistic with capacity C_i removed, to isolate the incremental value of C_i beyond controls
    x_noc = x.drop(columns=["potential_capacity_score"])
    rmse_noc = _cv_rmse("ridge", x_noc, y_arr, folds)
    obs_gain_c = rmse_noc - rmse_ridge
    null_gain_c = np.empty(N_PERM)
    for i in range(N_PERM):
        # permute capacity only (conditional permutation keeps controls-outcome structure intact)
        xp = x.copy()
        xp["potential_capacity_score"] = rng.permutation(xp["potential_capacity_score"].to_numpy())
        null_gain_c[i] = rmse_noc - _cv_rmse("ridge", xp, y_arr, folds)
    p_gain_c = float((np.sum(null_gain_c >= obs_gain_c) + 1) / (N_PERM + 1))
    return {
        "n_permutations": N_PERM,
        "seed": SEED,
        "capacity_vs_burden_spearman": float(obs),
        "capacity_vs_burden_perm_p": p_cap,
        "cv_rmse_intercept_only": rmse_int,
        "cv_rmse_ridge_full": rmse_ridge,
        "cv_rmse_gain_full_vs_intercept": float(obs_gain),
        "cv_rmse_gain_perm_p": p_gain,
        "cv_rmse_ridge_without_capacity": rmse_noc,
        "cv_rmse_gain_capacity_beyond_controls": float(obs_gain_c),
        "cv_rmse_gain_capacity_perm_p": p_gain_c,
        "note": "one-sided p-values for RMSE gains (H1: gain > 0); two-sided for the Spearman test; p = (count+1)/(N+1)",
    }


# ---------------------------------------------------------------------------------------------
# 2. bootstrap CIs for cross-evidence agreement / Spearman and model deltas
# ---------------------------------------------------------------------------------------------
def bootstrap_cross_evidence() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    variants = pd.read_csv(ROOT / "results/tables/stage4_evidence_variant_country_scores.csv")
    pairs = [
        ("baseline_owid_excess_mortality", "ghs_enhanced_owid"),
        ("baseline_owid_excess_mortality", "life_expectancy_loss"),
        ("ghs_enhanced_owid", "life_expectancy_loss"),
    ]
    rows = []
    payload = {}
    for a, b in pairs:
        left = variants[variants.evidence_variant.eq(a)].set_index("iso3")
        right = variants[variants.evidence_variant.eq(b)].set_index("iso3")
        common = left.index.intersection(right.index)
        pa = left.loc[common, "conversion_profile"].to_numpy()
        pb = right.loc[common, "conversion_profile"].to_numpy()
        ea = left.loc[common, "conversion_efficiency_zscore"].to_numpy()
        eb = right.loc[common, "conversion_efficiency_zscore"].to_numpy()
        n = len(common)
        agree = float(np.mean(pa == pb))
        rho = float(stats.spearmanr(ea, eb).statistic)
        ag_b = np.empty(N_BOOT)
        rho_b = np.empty(N_BOOT)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            ag_b[i] = np.mean(pa[idx] == pb[idx])
            rho_b[i] = stats.spearmanr(ea[idx], eb[idx]).statistic
        # Cohen's kappa as a chance-corrected companion to raw agreement
        cats = sorted(set(pa) | set(pb))
        pe = sum(np.mean(pa == c) * np.mean(pb == c) for c in cats)
        kappa = (agree - pe) / (1 - pe) if pe < 1 else np.nan
        row = {
            "evidence_variant_a": a,
            "evidence_variant_b": b,
            "n": n,
            "profile_agreement": agree,
            "profile_agreement_ci_low": float(np.percentile(ag_b, 2.5)),
            "profile_agreement_ci_high": float(np.percentile(ag_b, 97.5)),
            "cohen_kappa": float(kappa),
            "spearman": rho,
            "spearman_ci_low": float(np.percentile(rho_b, 2.5)),
            "spearman_ci_high": float(np.percentile(rho_b, 97.5)),
            "spearman_p": float(stats.spearmanr(ea, eb).pvalue),
        }
        rows.append(row)
        payload[f"{a}__vs__{b}"] = row
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cross_evidence_bootstrap_ci.csv", index=False)
    payload["n_boot"] = N_BOOT
    payload["method"] = "country-level percentile bootstrap of the paired 106-country set, conditional on the fitted labels/scores"
    return out, payload


def bootstrap_model_deltas(n_boot: int = 1000) -> dict:
    """Country-level percentile bootstrap CI for the CV-RMSE difference (alternative - ridge).

    Each resample draws 106 countries with replacement, runs one 5-fold CV (same folds for all models)
    and records RMSE(alternative) - RMSE(ridge). Elastic-net hyper-parameters are fixed at the values
    selected by ElasticNetCV on the full sample so that 1,000 resamples stay within the time budget.
    """

    from joblib import Parallel, delayed
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline

    from revision_analysis.common import numeric_scaler

    x, y, work = model_data()
    y_arr = y.to_numpy()
    enet_cv = make_model("elastic_net", x).fit(x, y_arr)
    alpha = float(enet_cv.named_steps["model"].alpha_)
    l1 = float(enet_cv.named_steps["model"].l1_ratio_)
    models = ["linear", "huber", "elastic_net", "random_forest", "gradient_boosting"]

    def build(name: str, xx: pd.DataFrame):
        if name == "elastic_net":
            return Pipeline([("scale", numeric_scaler(xx)), ("model", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=20000))])
        return make_model(name, xx)

    def one(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(y_arr), len(y_arr))
        xb = x.iloc[idx].reset_index(drop=True)
        yb = y_arr[idx]
        # Grouped folds: every copy of a resampled country goes to the same fold, otherwise duplicated
        # countries would sit in both the training and the test fold and flexible models would memorize them.
        fold_of_original = rng.permutation(np.arange(len(y_arr)) % 5)
        fold_id = fold_of_original[idx]
        folds = [(np.where(fold_id != k)[0], np.where(fold_id == k)[0]) for k in range(5) if np.any(fold_id == k)]
        out = {}
        err_ridge = np.empty(len(yb))
        for tr, te in folds:
            m = build("ridge", xb).fit(xb.iloc[tr], yb[tr])
            err_ridge[te] = yb[te] - m.predict(xb.iloc[te])
        rmse_ridge = float(np.sqrt(np.mean(err_ridge**2)))
        for name in models:
            err = np.empty(len(yb))
            for tr, te in folds:
                m = build(name, xb).fit(xb.iloc[tr], yb[tr])
                err[te] = yb[te] - m.predict(xb.iloc[te])
            out[name] = float(np.sqrt(np.mean(err**2))) - rmse_ridge
        return out

    draws = Parallel(n_jobs=-1)(delayed(one)(SEED + i) for i in range(n_boot))
    result = {}
    for name in models:
        vals = np.array([d[name] for d in draws])
        result[name] = {
            "delta_rmse_mean": float(vals.mean()),
            "ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "share_resamples_alternative_better": float(np.mean(vals < 0)),
        }
    result["n_boot"] = n_boot
    result["elastic_net_fixed_alpha"] = alpha
    result["elastic_net_fixed_l1_ratio"] = l1
    result["note"] = "country-level percentile bootstrap (resample 106 countries with replacement; one 5-fold CV per resample with all copies of a country in the same fold; identical folds across models); positive delta = worse than ridge"
    return result


# ---------------------------------------------------------------------------------------------
# 3. calibration of the ridge expected-burden model
# ---------------------------------------------------------------------------------------------
def calibration() -> tuple[pd.DataFrame, dict]:
    x, y, work = model_data()
    y_arr = y.to_numpy()
    oof = np.load(RESULTS_DIR / "model_comparison_oof_ridge_by_repeat.npy")  # repeats x n
    pred = oof.mean(axis=0)
    slope, intercept, r, p, se = stats.linregress(pred, y_arr)
    # cross-conformal residual-quantile intervals: for country i in outer fold k, the half-width is the
    # empirical quantile of |OOF residual| among countries NOT in fold k (first CV repeat's folds).
    from sklearn.model_selection import RepeatedKFold

    rkf = RepeatedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    first_repeat = [te for _, (tr, te) in zip(range(5), rkf.split(x))]
    resid = np.abs(y_arr - oof[0])
    cover = {}
    width = {}
    lo80 = np.empty(len(y_arr)); hi80 = np.empty(len(y_arr)); lo95 = np.empty(len(y_arr)); hi95 = np.empty(len(y_arr))
    for te in first_repeat:
        mask = np.ones(len(y_arr), bool); mask[te] = False
        q80 = np.quantile(resid[mask], 0.80); q95 = np.quantile(resid[mask], 0.95)
        lo80[te] = oof[0][te] - q80; hi80[te] = oof[0][te] + q80
        lo95[te] = oof[0][te] - q95; hi95[te] = oof[0][te] + q95
    cover["80"] = float(np.mean((y_arr >= lo80) & (y_arr <= hi80)))
    cover["95"] = float(np.mean((y_arr >= lo95) & (y_arr <= hi95)))
    width["80"] = float(np.mean(hi80 - lo80)); width["95"] = float(np.mean(hi95 - lo95))
    cal = work[["iso3", "country", "observed_shock_burden"]].copy()
    cal["oof_predicted_ridge_mean_over_repeats"] = pred
    cal["oof_predicted_ridge_repeat1"] = oof[0]
    cal["pi80_low"] = lo80; cal["pi80_high"] = hi80; cal["pi95_low"] = lo95; cal["pi95_high"] = hi95
    cal["covered_80"] = (y_arr >= lo80) & (y_arr <= hi80)
    cal["covered_95"] = (y_arr >= lo95) & (y_arr <= hi95)
    cal.to_csv(RESULTS_DIR / "calibration.csv", index=False)
    # calibration by decile of predicted burden
    cal["decile"] = pd.qcut(cal["oof_predicted_ridge_mean_over_repeats"], 5, labels=False)
    dec = cal.groupby("decile").agg(mean_pred=("oof_predicted_ridge_mean_over_repeats", "mean"), mean_obs=("observed_shock_burden", "mean"), n=("iso3", "size")).reset_index()
    dec.to_csv(RESULTS_DIR / "calibration_bins.csv", index=False)
    payload = {
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_slope_se": float(se),
        "calibration_slope_ci_low": float(slope - 1.96 * se),
        "calibration_slope_ci_high": float(slope + 1.96 * se),
        "oof_pearson_r": float(r),
        "oof_r2": float(1 - np.sum((y_arr - pred) ** 2) / np.sum((y_arr - y_arr.mean()) ** 2)),
        "oof_rmse": float(np.sqrt(np.mean((y_arr - pred) ** 2))),
        "coverage_80": cover["80"],
        "coverage_95": cover["95"],
        "mean_width_80": width["80"],
        "mean_width_95": width["95"],
        "interval_method": "cross-conformal residual-quantile intervals: half-width = 80%/95% quantile of |out-of-fold residual| among countries outside the country's own fold (first CV repeat)",
        "n": int(len(y_arr)),
    }
    return cal, payload


# ---------------------------------------------------------------------------------------------
# 4. profile-difference sanity check
# ---------------------------------------------------------------------------------------------
def profile_difference() -> dict:
    prof = pd.read_csv(ROOT / "results/tables/conversion_profiles.csv")
    four = prof[prof.conversion_profile.ne("uncertain / data-limited systems")]
    groups = [g["conversion_efficiency_zscore"].to_numpy() for _, g in four.groupby("conversion_profile")]
    names = [n for n, _ in four.groupby("conversion_profile")]
    kw = stats.kruskal(*groups)
    pair_rows = []
    pvals = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u = stats.mannwhitneyu(groups[i], groups[j], alternative="two-sided")
            pair_rows.append({"profile_a": names[i], "profile_b": names[j], "n_a": len(groups[i]), "n_b": len(groups[j]), "mannwhitney_u": float(u.statistic), "p_raw": float(u.pvalue)})
            pvals.append(float(u.pvalue))
    # Holm correction
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    for row, a in zip(pair_rows, adj):
        row["p_holm"] = float(a)
    pd.DataFrame(pair_rows).to_csv(RESULTS_DIR / "profile_difference_pairwise.csv", index=False)
    return {
        "kruskal_h": float(kw.statistic),
        "kruskal_p": float(kw.pvalue),
        "n_profiles": len(names),
        "n_countries": int(len(four)),
        "pairwise": pair_rows,
        "max_holm_p": float(adj.max()),
        "note": "expected by construction (profiles are defined from E_i); reported as a sanity/reporting item only",
    }


def run() -> None:
    ensure_results_dirs()
    perm = permutation_tests()
    print("permutation:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in perm.items() if k != "note"})
    ce, ce_payload = bootstrap_cross_evidence()
    print(ce.round(3).to_string())
    deltas = bootstrap_model_deltas()
    print("deltas:", deltas)
    _, cal = calibration()
    print("calibration:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in cal.items() if k != "interval_method"})
    pdiff = profile_difference()
    print("kruskal:", pdiff["kruskal_h"], pdiff["kruskal_p"], "max holm p", pdiff["max_holm_p"])
    payload = {"permutation": perm, "cross_evidence_bootstrap": ce_payload, "model_delta_bootstrap": deltas, "calibration": cal, "profile_difference": pdiff}
    (RESULTS_DIR / "stat_tests.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    update_numbers("statistics", payload)


def rerun_deltas_only() -> None:
    path = RESULTS_DIR / "stat_tests.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model_delta_bootstrap"] = bootstrap_model_deltas()
    path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    update_numbers("statistics", payload)
    print(payload["model_delta_bootstrap"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--deltas-only":
        rerun_deltas_only()
    else:
        run()
