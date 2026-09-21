"""A0: collect the headline numbers of the (corrected) base pipeline into revision_numbers.json."""

from __future__ import annotations

import pandas as pd

from revision_analysis.common import ROOT, ensure_results_dirs, update_numbers


def run() -> dict:
    ensure_results_dirs()
    R = ROOT / "results/tables"
    potential = pd.read_csv(R / "potential_capacity_scores.csv")
    profiles = pd.read_csv(R / "conversion_profiles.csv")
    meta = pd.read_csv(ROOT / "data/processed/indicator_metadata_expanded.csv")
    feat = meta[meta["use_as_feature_in_expanded_score"].astype(bool)]
    ghs = pd.read_csv(R / "stage4_ghs_2019_integration_audit.csv").iloc[0]
    ce = pd.read_csv(R / "stage4_cross_evidence_profile_stability.csv")
    lab = pd.read_csv(R / "stage4_country_label_stability.csv")
    bott = pd.read_csv(R / "stage4_bottleneck_cross_evidence_stability.csv")
    mc = pd.read_csv(R / "conversion_monte_carlo_summary.csv")
    boot = pd.read_csv(R / "conversion_efficiency_uncertainty.csv")
    stable = pd.read_csv(ROOT / "paper_assets/tables/table8_stable_case_audit.csv")
    diag = pd.read_csv(R / "expected_shock_burden_model_diagnostics.csv")
    counts = profiles.groupby("conversion_profile")["conversion_efficiency_zscore"].agg(["count", "median"])
    cat = lab["country_label_category"]
    out = {
        "n_scoring_countries": int(potential["iso3"].nunique()),
        "n_owid_validation_countries": int(profiles["iso3"].nunique()),
        "n_indicators": int(len(feat)),
        "indicators_per_block": feat.groupby("block").size().to_dict(),
        "n_ghs_matched_countries": int(ghs["n_matched_iso3_countries"]),
        "n_life_expectancy_countries": int(pd.read_csv(R / "stage4_life_expectancy_coverage.csv")["coverage_count"].max()),
        "stage2_single_cv": {r["model_version"]: {"cv_rmse": r["cv_rmse"], "in_sample_r2": r["r2"], "in_sample_adj_r2": r["adjusted_r2"]} for _, r in diag.iterrows()},
        "profile_counts": {k: int(v) for k, v in counts["count"].items()},
        "profile_median_efficiency": {k: float(v) for k, v in counts["median"].items()},
        "cross_evidence": {
            f"{r.evidence_variant_a}__vs__{r.evidence_variant_b}": {
                "n": int(r.n),
                "profile_agreement": float(r.profile_agreement),
                "spearman": float(r.conversion_efficiency_spearman),
                "stable_label_count": int(r.stable_label_count),
                "unstable_label_count": int(r.unstable_label_count),
            }
            for r in ce.itertuples()
        },
        "bottleneck_agreement": {f"{r.evidence_variant_a}__vs__{r.evidence_variant_b}": float(r.bottleneck_block_agreement) for r in bott.itertuples()},
        "label_triage": {
            "n_labelled": int(len(lab)),
            "n_scoring_without_any_lens": int((~potential["iso3"].isin(lab["iso3"])).sum()),
            "scoring_without_any_lens": potential.loc[~potential["iso3"].isin(lab["iso3"]), "country"].tolist(),
            "stable": int(cat.str.startswith("stable").sum()),
            "evidence_dependent": int((cat == "evidence-dependent label").sum()),
            "data_limited": int((cat == "data-limited").sum()),
            "stable_by_profile": cat[cat.str.startswith("stable")].value_counts().to_dict(),
        },
        "monte_carlo_robust_profile_share": float((mc["robustness_category"] == "robust").mean()),
        "monte_carlo_mean_profile_stability": float(mc["profile_stability_score"].mean()),
        "bootstrap_median_profile_stability": float(boot["profile_stability_score"].median()) if "profile_stability_score" in boot.columns else None,
        "stable_examples": {
            cat_name: grp["country"].tolist()
            for cat_name, grp in stable[stable["country_label_category"].str.startswith("stable")].groupby("country_label_category")
        },
        "evidence_dependent_examples": stable[stable["country_label_category"].eq("evidence-dependent label")]["country"].tolist(),
        "data_limited_examples": stable[stable["country_label_category"].eq("data-limited")]["country"].tolist(),
    }
    update_numbers("pipeline", out)
    return out


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=1, default=str)[:4000])
