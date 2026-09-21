from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALID = {
    "effective converters",
    "capacity under-realizers",
    "adaptive over-performers",
    "structurally vulnerable systems",
    "uncertain / data-limited systems",
}


def test_conversion_profile_labels_are_valid() -> None:
    path = ROOT / "results/tables/conversion_profiles.csv"
    if not path.exists():
        return
    profiles = pd.read_csv(path)
    assert set(profiles["conversion_profile"]).issubset(VALID)
    assert profiles["profile_rule"].notna().all()

