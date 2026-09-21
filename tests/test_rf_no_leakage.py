from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.rf_benchmark_audit import FORBIDDEN_TERMS, RF_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def test_rf_features_exclude_outcome_and_postshock_terms() -> None:
    joined = " ".join(RF_FEATURES).lower()
    assert not any(term in joined for term in FORBIDDEN_TERMS)


def test_rf_audit_reports_no_leakage_after_run() -> None:
    path = ROOT / "results/tables/rf_benchmark_audit.csv"
    if not path.exists():
        return
    audit = pd.read_csv(path)
    assert audit["no_leakage_detected"].astype(bool).all()

