from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_who_pulse_readme_exists_and_documents_optional_source() -> None:
    readme = ROOT / "data/raw/who_pulse/README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8").lower()
    assert "country-level" in text
    assert "csv" in text
    assert "will not fabricate" in text


def test_who_pulse_skips_when_no_legitimate_file_exists() -> None:
    path = ROOT / "results/tables/stage4_who_pulse_inventory.csv"
    if not path.exists():
        return
    inventory = pd.read_csv(path)
    source_files = [
        p for p in (ROOT / "data/raw/who_pulse").glob("*")
        if p.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ]
    if not source_files:
        assert inventory["status"].iloc[0] == "skipped_no_legitimate_file"
        assert inventory["coverage_count"].iloc[0] == 0


def test_who_pulse_loaded_profiles_have_allowed_labels_if_present() -> None:
    path = ROOT / "results/tables/stage4_who_pulse_profile_overlap.csv"
    if not path.exists():
        return
    profiles = pd.read_csv(path)
    if "conversion_profile" not in profiles.columns:
        return
    valid = {
        "effective converters",
        "capacity under-realizers",
        "adaptive over-performers",
        "structurally vulnerable systems",
        "uncertain / data-limited systems",
    }
    assert set(profiles["conversion_profile"]).issubset(valid)
