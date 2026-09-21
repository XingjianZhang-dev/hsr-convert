"""A3: transparent static-index baselines compared with HSR-Convert conversion efficiency [R2-6]."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision_analysis.common import ROOT, RESULTS_DIR, SEED, baseline_profiles, ensure_results_dirs, update_numbers

N_BOOT = 2000


def _boot_spearman(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(a)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        vals[i] = spearmanr(a[idx], b[idx]).statistic
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _overlap(rank_a: pd.Series, rank_b: pd.Series, k: int, top: bool) -> float:
    sa = set(rank_a.nsmallest(k).index) if top else set(rank_a.nlargest(k).index)
    sb = set(rank_b.nsmallest(k).index) if top else set(rank_b.nlargest(k).index)
    return len(sa & sb) / k


def run() -> pd.DataFrame:
    ensure_results_dirs()
    rng = np.random.default_rng(SEED)
    prof = baseline_profiles(ROOT)
    ghs = pd.read_csv(ROOT / "results/tables/stage4_preparedness_block_with_ghs.csv")[["iso3", "ghs_overall_2019"]]
    df = prof.merge(ghs, on="iso3", how="left").set_index("iso3")
    df["outcome_only_score"] = -df["observed_shock_burden"]  # lower burden = better
    baselines = {
        "ghs_2019_overall": ("GHS 2019 overall index (static preparedness ranking)", "ghs_overall_2019"),
        "capacity_only": ("HSR-Convert potential capacity C_i only (no outcome linkage)", "potential_capacity_score"),
        "outcome_only": ("Raw 2020-2022 excess deaths per million, sign-flipped (no capacity adjustment)", "outcome_only_score"),
    }
    rows = []
    disagreement_frames = []
    for key, (label, col) in baselines.items():
        sub = df.dropna(subset=[col, "conversion_efficiency_zscore"])
        e = sub["conversion_efficiency_zscore"].to_numpy()
        b = sub[col].to_numpy()
        rho = spearmanr(e, b)
        lo, hi = _boot_spearman(e, b, rng)
        # ranks: 1 = best (highest efficiency / highest index / lowest burden)
        rank_e = sub["conversion_efficiency_zscore"].rank(ascending=False, method="min")
        rank_b = sub[col].rank(ascending=False, method="min")
        row = {
            "baseline": key,
            "description": label,
            "n": int(len(sub)),
            "spearman_with_conversion_efficiency": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
            "spearman_ci_low": lo,
            "spearman_ci_high": hi,
            "top10_overlap": _overlap(rank_e, rank_b, 10, True),
            "top20_overlap": _overlap(rank_e, rank_b, 20, True),
            "bottom10_overlap": _overlap(rank_e, rank_b, 10, False),
            "bottom20_overlap": _overlap(rank_e, rank_b, 20, False),
        }
        # association of the baseline itself with observed burden (the preparedness-outcome literature check)
        if key != "outcome_only":
            r2 = spearmanr(sub[col], sub["observed_shock_burden"])
            lo2, hi2 = _boot_spearman(sub[col].to_numpy(), sub["observed_shock_burden"].to_numpy(), rng)
            row.update({"spearman_with_observed_burden": float(r2.statistic), "spearman_with_observed_burden_p": float(r2.pvalue), "spearman_with_observed_burden_ci_low": lo2, "spearman_with_observed_burden_ci_high": hi2})
        rows.append(row)
        d = sub[["country", "conversion_profile", "potential_capacity_score", "observed_shock_burden", "conversion_efficiency_zscore"]].copy()
        d["baseline_score"] = sub[col]
        d["baseline"] = key
        d["baseline_rank"] = rank_b.astype(int)
        d["efficiency_rank"] = rank_e.astype(int)
        d["rank_gap_efficiency_minus_baseline"] = d["efficiency_rank"] - d["baseline_rank"]
        d["abs_rank_gap"] = d["rank_gap_efficiency_minus_baseline"].abs()
        disagreement_frames.append(d.reset_index())
    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "baseline_comparison.csv", index=False)
    dis = pd.concat(disagreement_frames, ignore_index=True)
    # label stability for context
    stab = pd.read_csv(ROOT / "results/tables/stage4_country_label_stability.csv")[["iso3", "country_label_category"]]
    dis = dis.merge(stab, on="iso3", how="left")
    top = dis.sort_values(["baseline", "abs_rank_gap"], ascending=[True, False]).groupby("baseline").head(12)
    top.to_csv(RESULTS_DIR / "disagreement_cases.csv", index=False)
    payload = {r["baseline"]: {k: v for k, v in r.items() if k != "baseline"} for r in rows}
    payload["n_boot"] = N_BOOT
    payload["seed"] = SEED
    payload["ghs_coverage_in_validation_set"] = int(df["ghs_overall_2019"].notna().sum())
    # named disagreement examples: high GHS (top quartile) but negative efficiency, and low GHS (bottom quartile) but positive efficiency
    g = dis[dis.baseline.eq("ghs_2019_overall")].copy()
    q75, q25 = g["baseline_score"].quantile(0.75), g["baseline_score"].quantile(0.25)
    high_ghs_low_eff = g[(g.baseline_score >= q75) & (g.conversion_efficiency_zscore < 0)].sort_values("conversion_efficiency_zscore")
    low_ghs_high_eff = g[(g.baseline_score <= q25) & (g.conversion_efficiency_zscore > 0)].sort_values("conversion_efficiency_zscore", ascending=False)
    payload["ghs_top_quartile_negative_efficiency"] = high_ghs_low_eff[["iso3", "country", "baseline_score", "baseline_rank", "efficiency_rank", "conversion_efficiency_zscore", "conversion_profile", "country_label_category"]].to_dict("records")
    payload["ghs_bottom_quartile_positive_efficiency"] = low_ghs_high_eff[["iso3", "country", "baseline_score", "baseline_rank", "efficiency_rank", "conversion_efficiency_zscore", "conversion_profile", "country_label_category"]].to_dict("records")
    payload["n_ghs_top_quartile_negative_efficiency"] = int(len(high_ghs_low_eff))
    payload["n_ghs_top_quartile"] = int((g.baseline_score >= q75).sum())
    payload["n_ghs_bottom_quartile_positive_efficiency"] = int(len(low_ghs_high_eff))
    payload["n_ghs_bottom_quartile"] = int((g.baseline_score <= q25).sum())
    update_numbers("baselines", payload)
    return comparison


if __name__ == "__main__":
    print(run().round(3).to_string())
