from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_every_expanded_feature_has_block_assignment() -> None:
    cfg = yaml.safe_load((ROOT / "config/indicator_blocks.yml").read_text(encoding="utf-8"))
    valid_blocks = set(cfg["blocks"])
    for indicator in cfg["indicators"]:
        assert indicator.get("block") in valid_blocks, indicator["variable"]
        assert indicator.get("direction") in {"beneficial", "cost"}, indicator["variable"]


def test_expanded_metadata_preserves_block_assignments() -> None:
    path = ROOT / "data/processed/indicator_metadata_expanded.csv"
    if not path.exists():
        return
    meta = pd.read_csv(path)
    features = meta[meta["use_as_feature_in_expanded_score"].astype(bool)]
    assert not features.empty
    assert features["block"].notna().all()
    assert set(features["block"]).issuperset({"capacity", "preparedness", "vulnerability", "equity_access"})

