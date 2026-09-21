from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_stable_case_audit_has_required_columns() -> None:
    path = ROOT / "results/tables/stage4_stable_case_audit.csv"
    if not path.exists():
        return
    audit = pd.read_csv(path)
    required = {
        "iso3",
        "country",
        "profile_label",
        "profile_stability_probability",
        "conversion_efficiency_zscore",
        "conversion_efficiency_ci_low",
        "conversion_efficiency_ci_high",
        "observed_shock_burden",
        "expected_shock_burden",
        "realized_resilience_gap",
        "potential_capacity_score",
        "capacity_block_score",
        "preparedness_block_score",
        "vulnerability_block_score",
        "equity_access_block_score",
        "main_bottleneck_block",
        "bottleneck_probability",
        "missingness_rate",
        "validation_outcomes_available",
        "country_label_category",
        "diagnostic_note",
        "n",
        "coverage_count",
        "missing_count",
        "missing_rate",
    }
    assert required.issubset(audit.columns)
    assert not audit.empty


def test_stable_case_notes_are_descriptive_not_prescriptive() -> None:
    path = ROOT / "results/tables/stage4_stable_case_audit.csv"
    if not path.exists():
        return
    audit = pd.read_csv(path)
    notes = " ".join(audit["diagnostic_note"].astype(str)).lower()
    assert "descriptive only" in notes
    assert "recommend" not in notes
    assert "should implement" not in notes
