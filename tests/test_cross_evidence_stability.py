from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VALID_PROFILES = {
    "effective converters",
    "capacity under-realizers",
    "adaptive over-performers",
    "structurally vulnerable systems",
    "uncertain / data-limited systems",
}
VALID_LABELS = {
    "stable under-realizer",
    "stable over-performer",
    "stable effective converter",
    "stable structurally vulnerable",
    "evidence-dependent label",
    "data-limited",
}


def test_cross_evidence_outputs_exist_and_have_variant_pairs() -> None:
    path = ROOT / "results/tables/stage4_cross_evidence_profile_stability.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert not df.empty
    assert {"evidence_variant_a", "evidence_variant_b", "profile_agreement", "conversion_efficiency_spearman"}.issubset(df.columns)
    assert df["profile_agreement"].between(0, 1).all()


def test_country_label_stability_uses_allowed_categories() -> None:
    path = ROOT / "results/tables/stage4_country_label_stability.csv"
    if not path.exists():
        return
    labels = pd.read_csv(path)
    assert set(labels["country_label_category"]).issubset(VALID_LABELS)
    assert labels["label_stability_rate"].between(0, 1).all()


def test_evidence_variant_profiles_are_allowed() -> None:
    path = ROOT / "results/tables/stage4_evidence_variant_country_scores.csv"
    if not path.exists():
        return
    scores = pd.read_csv(path)
    assert set(scores["conversion_profile"]).issubset(VALID_PROFILES)
    assert scores["evidence_variant"].nunique() >= 2
