from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_feature_years_are_pre_shock_only() -> None:
    for config_name in ["indicators.yml", "indicator_blocks.yml"]:
        cfg = yaml.safe_load((ROOT / f"config/{config_name}").read_text(encoding="utf-8"))
        for indicator in cfg["indicators"]:
            if indicator.get("use_as_feature", False):
                assert max(indicator["years_used"]) < 2020, indicator["variable"]


def test_no_covid_response_or_outcome_terms_in_features() -> None:
    forbidden = [
        "covid",
        "sars",
        "death",
        "deaths",
        "case",
        "cases",
        "testing",
        "vaccination",
        "vaccine",
        "stringency",
        "lockdown",
        "excess",
    ]
    for config_name in ["indicators.yml", "indicator_blocks.yml"]:
        cfg = yaml.safe_load((ROOT / f"config/{config_name}").read_text(encoding="utf-8"))
        for indicator in cfg["indicators"]:
            if not indicator.get("use_as_feature", False):
                continue
            text = " ".join(
                str(indicator.get(field, "")).lower()
                for field in ["variable", "label", "source_code", "conceptual_dimension"]
            )
            assert not any(term in text for term in forbidden), indicator["variable"]


def test_outcomes_not_in_potential_capacity_outputs() -> None:
    path = ROOT / "results/tables/potential_capacity_scores.csv"
    if not path.exists():
        return
    header = path.read_text(encoding="utf-8").splitlines()[0].lower()
    forbidden = ["excess", "death", "deaths", "outcome", "shock_burden", "covid"]
    assert not any(term in header for term in forbidden)
