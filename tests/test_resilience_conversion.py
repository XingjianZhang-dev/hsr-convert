from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_conversion_tables_contain_required_columns() -> None:
    path = ROOT / "results/tables/conversion_efficiency_scores.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    required = {
        "iso3",
        "potential_capacity_score",
        "observed_shock_burden",
        "predicted_shock_burden",
        "realized_resilience_gap",
        "conversion_efficiency_zscore",
        "conversion_efficiency_percentile",
        "higher_efficiency_is_better",
    }
    assert required.issubset(df.columns)
    assert df["higher_efficiency_is_better"].astype(bool).all()


def test_higher_conversion_efficiency_means_better_than_expected() -> None:
    path = ROOT / "results/tables/conversion_efficiency_scores.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    top = df.nlargest(10, "conversion_efficiency_zscore")["realized_resilience_gap"].median()
    bottom = df.nsmallest(10, "conversion_efficiency_zscore")["realized_resilience_gap"].median()
    assert top < bottom

