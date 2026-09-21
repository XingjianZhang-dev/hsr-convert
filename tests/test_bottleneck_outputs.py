from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_bottleneck_outputs_have_required_columns() -> None:
    path = ROOT / "results/tables/conversion_bottleneck_scores.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    required = {
        "iso3",
        "conversion_profile",
        "weakest_bottleneck_block",
        "weakest_bottleneck_gap",
        "conversion_efficiency_zscore",
        "realized_resilience_gap",
        "n_validation_countries",
    }
    assert required.issubset(df.columns)
    assert df["weakest_bottleneck_block"].notna().all()
