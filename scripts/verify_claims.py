"""Lightweight check that the shipped result tables carry the numbers reported in the manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read_csv(relative_path: str) -> pd.DataFrame:
    path = ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing required asset: {relative_path}")
    return pd.read_csv(path)


def assert_close(actual: float, expected: float, tolerance: float = 1e-3) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"Expected {expected}, got {actual}")


def main() -> None:
    potential = read_csv("results/tables/potential_capacity_scores.csv")
    baseline = read_csv("results/tables/conversion_profiles.csv")
    ghs_audit = read_csv("results/tables/stage4_ghs_2019_integration_audit.csv")
    labels = read_csv("results/tables/stage4_country_label_stability.csv")
    agreement = read_csv("results/tables/stage4_cross_evidence_profile_stability.csv")
    diagnostics = read_csv("results/tables/expected_shock_burden_model_diagnostics.csv")
    table7 = read_csv("paper_assets/tables/table7_cross_evidence_stability.csv")

    assert len(potential) == 193, f"Expected 193 scoring countries, got {len(potential)}"
    assert len(baseline) == 106, f"Expected 106 OWID validation countries, got {len(baseline)}"
    counts = baseline["conversion_profile"].value_counts()
    expected_counts = {
        "adaptive over-performers": 20,
        "capacity under-realizers": 23,
        "effective converters": 20,
        "structurally vulnerable systems": 16,
        "uncertain / data-limited systems": 27,
    }
    for profile, n in expected_counts.items():
        assert int(counts.get(profile, 0)) == n, f"Expected {n} {profile}, got {counts.get(profile, 0)}"
    ridge = diagnostics[diagnostics["model_version"].eq("ridge")].iloc[0]
    assert_close(float(ridge["cv_rmse"]), 1764.982, tolerance=0.01)

    ghs_row = ghs_audit.iloc[0]
    assert ghs_row["status"] == "loaded", f"Expected GHS status loaded, got {ghs_row['status']}"
    assert int(ghs_row["n_matched_iso3_countries"]) == 189

    cat = labels["country_label_category"]
    stable = int(cat.str.startswith("stable").sum())
    evidence_dependent = int((cat == "evidence-dependent label").sum())
    data_limited = int((cat == "data-limited").sum())
    assert stable == 41, f"Expected 41 stable labels, got {stable}"
    assert evidence_dependent == 38, f"Expected 38 evidence-dependent labels, got {evidence_dependent}"
    assert data_limited == 107, f"Expected 107 data-limited labels, got {data_limited}"

    pairs = {
        ("baseline_owid_excess_mortality", "ghs_enhanced_owid"): (0.971698, 0.996040),
        ("baseline_owid_excess_mortality", "life_expectancy_loss"): (0.641509, 0.660448),
        ("ghs_enhanced_owid", "life_expectancy_loss"): (0.650943, 0.654150),
    }
    for (variant_a, variant_b), (expected_profile, expected_spearman) in pairs.items():
        row = agreement[(agreement["evidence_variant_a"] == variant_a) & (agreement["evidence_variant_b"] == variant_b)]
        if row.empty:
            raise AssertionError(f"Missing cross-evidence pair: {variant_a} vs {variant_b}")
        item = row.iloc[0]
        assert_close(float(item["profile_agreement"]), expected_profile)
        assert_close(float(item["conversion_efficiency_spearman"]), expected_spearman)
    assert len(table7) == 3, f"Expected 3 Table 7 rows, got {len(table7)}"

    revision = ROOT / "results/revision/revision_numbers.json"
    if revision.exists():
        numbers = json.loads(revision.read_text(encoding="utf-8"))
        thr = numbers["sensitivity"]["thresholds"]
        assert thr["n_countries_invariant_main_grid"] == 89, thr["n_countries_invariant_main_grid"]
        assert_close(numbers["model_comparison"]["ridge"]["cv_rmse_mean"], 1759.227, tolerance=0.01)
        print("Revision analysis numbers: OK (89 threshold-invariant countries; ridge repeated-CV RMSE 1759.2)")

    print("Verification passed.")
    print(f"Scoring countries: {len(potential)}; OWID validation countries: {len(baseline)}; GHS matched: 189")
    print(f"Labels: stable={stable}, evidence-dependent={evidence_dependent}, data-limited={data_limited}")


if __name__ == "__main__":
    main()
