from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTCOME_COLS = {
    "life_expectancy_loss_2019_2020",
    "life_expectancy_loss_2019_2021",
    "life_expectancy_loss_2019_2022",
    "max_life_expectancy_drop_2020_2022",
}


def test_life_expectancy_loss_outcomes_are_validation_only() -> None:
    outcome_path = ROOT / "data/processed/life_expectancy_loss_outcomes.csv"
    if not outcome_path.exists():
        return
    outcomes = pd.read_csv(outcome_path)
    assert OUTCOME_COLS.issubset(outcomes.columns)
    assert outcomes["higher_is_worse"].astype(bool).all()
    assert outcomes["source_indicator"].eq("SP.DYN.LE00.IN").all()

    potential_header = (ROOT / "results/tables/potential_capacity_scores.csv").read_text(encoding="utf-8").splitlines()[0]
    forbidden = OUTCOME_COLS | {"life_expectancy_2020", "life_expectancy_2021", "life_expectancy_2022"}
    assert not any(term in potential_header for term in forbidden)


def test_life_expectancy_validation_outputs_have_coverage_and_source() -> None:
    path = ROOT / "results/tables/stage4_life_expectancy_secondary_validation.csv"
    if not path.exists():
        return
    validation = pd.read_csv(path)
    assert set(validation["outcome"]).issubset(OUTCOME_COLS)
    assert {"n", "coverage_count", "missing_count", "missing_rate"}.issubset(validation.columns)
    assert validation["source_indicator"].eq("SP.DYN.LE00.IN").all()


def test_life_expectancy_profiles_use_allowed_labels() -> None:
    path = ROOT / "results/tables/stage4_life_expectancy_conversion_profiles.csv"
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
    assert profiles["higher_is_worse"].astype(bool).all()
