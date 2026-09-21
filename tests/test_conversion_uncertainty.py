from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_conversion_uncertainty_contains_intervals_and_probabilities() -> None:
    path = ROOT / "results/tables/conversion_efficiency_uncertainty.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    required = {
        "conversion_efficiency_mean",
        "conversion_efficiency_ci_low",
        "conversion_efficiency_ci_high",
        "probability_under_realizer",
        "probability_over_performer",
        "profile_stability_score",
    }
    assert required.issubset(df.columns)
    assert (df["conversion_efficiency_ci_low"] <= df["conversion_efficiency_ci_high"]).all()
    assert df["profile_stability_score"].between(0, 1).all()


def test_monte_carlo_profile_probabilities_sum_to_one() -> None:
    path = ROOT / "results/tables/profile_probability_table.csv"
    if not path.exists():
        return
    probs = pd.read_csv(path).groupby("iso3")["profile_probability"].sum()
    assert ((probs - 1.0).abs() < 1e-6).all()

