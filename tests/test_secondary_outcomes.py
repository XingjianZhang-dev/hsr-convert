from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_secondary_outcome_inventory_does_not_fabricate_missing_sources() -> None:
    path = ROOT / "results/tables/secondary_outcome_inventory.csv"
    if not path.exists():
        return
    inventory = pd.read_csv(path)
    if inventory["status"].iloc[0].startswith("skipped"):
        assert inventory["coverage_count"].max() == 0
        assert "No" in inventory["note"].iloc[0] or "no" in inventory["note"].iloc[0]

