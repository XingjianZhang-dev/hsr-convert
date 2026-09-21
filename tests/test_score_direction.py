from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_score_rank_direction_audit_documents_directions() -> None:
    path = ROOT / "results/tables/score_rank_direction_audit.csv"
    if not path.exists():
        return
    audit = pd.read_csv(path)
    required = {
        "median_score",
        "median_rank",
        "mean_rank",
        "top_quartile_probability",
        "bottom_quartile_probability",
        "rank_instability_index",
        "resilience_score_rank_based",
    }
    assert required.issubset(set(audit["score_variable"]))
    assert audit["higher_value_interpretation"].notna().all()
    assert audit["n"].min() > 0

