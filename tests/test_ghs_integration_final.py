from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_stage4_ghs_skips_absent_file_without_fabrication() -> None:
    audit_path = ROOT / "results/tables/stage4_ghs_2019_integration_audit.csv"
    if not audit_path.exists():
        return
    audit = pd.read_csv(audit_path)
    source_file = ROOT / "data/raw/ghs/ghs_index_2019.csv"
    if not source_file.exists():
        assert audit["status"].iloc[0] == "skipped_missing_file"
        assert audit["coverage_count"].iloc[0] == 0
        assert audit["file_hash_sha256"].fillna("").iloc[0] == ""
        assert "no values fabricated" in audit["source_provenance"].iloc[0]


def test_stage4_ghs_loaded_file_has_provenance_if_present() -> None:
    audit_path = ROOT / "results/tables/stage4_ghs_2019_integration_audit.csv"
    source_file = ROOT / "data/raw/ghs/ghs_index_2019.csv"
    if not audit_path.exists() or not source_file.exists():
        return
    audit = pd.read_csv(audit_path)
    if audit["status"].iloc[0] == "skipped_missing_file" and source_file.stat().st_mtime > audit_path.stat().st_mtime:
        return
    assert audit["status"].iloc[0] == "loaded"
    assert audit["file_hash_sha256"].astype(str).str.len().iloc[0] == 64
    assert audit["n_matched_iso3_countries"].iloc[0] > 0
    assert audit["detected_columns"].astype(str).str.len().iloc[0] > 0
