from __future__ import annotations

import pandas as pd

from src.mcdm_methods import METHODS, rank_scores
from src.weighting import compute_weights


def test_dominated_alternative_is_not_ranked_first() -> None:
    matrix = pd.DataFrame(
        {
            "capacity": [0.2, 0.8, 0.5],
            "coverage": [0.2, 0.8, 0.5],
            "financing": [0.2, 0.8, 0.5],
        },
        index=["dominated", "dominant", "middle"],
    )
    weights = pd.Series(1 / 3, index=matrix.columns)
    for method in METHODS:
        ranked = rank_scores(matrix, weights, method)
        assert ranked.loc["dominant", "rank"] == 1, method
        assert ranked.loc["dominated", "rank"] > ranked.loc["dominant", "rank"], method


def test_weight_methods_sum_to_one() -> None:
    matrix = pd.DataFrame(
        {
            "a": [0.1, 0.5, 0.9, 0.3],
            "b": [0.8, 0.7, 0.2, 0.4],
            "c": [0.3, 0.4, 0.7, 0.6],
        }
    )
    for method in ["equal", "std", "entropy", "critic", "merec"]:
        weights = compute_weights(matrix, method)
        assert (weights >= 0).all(), method
        assert abs(weights.sum() - 1.0) < 1e-9, method

