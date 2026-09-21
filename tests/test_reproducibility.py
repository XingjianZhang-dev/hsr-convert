from __future__ import annotations

import pandas as pd

from src.imputation import impute_matrix
from src.mcdm_methods import rank_scores
from src.normalization import normalize_matrix
from src.weighting import compute_weights


def test_repeated_pipeline_components_are_deterministic() -> None:
    raw = pd.DataFrame(
        {
            "benefit": [1.0, 2.0, None, 4.0],
            "cost": [8.0, None, 5.0, 3.0],
        },
        index=["a", "b", "c", "d"],
    )
    directions = {"benefit": "beneficial", "cost": "cost"}
    first = impute_matrix(raw, "iterative", random_state=123)
    second = impute_matrix(raw, "iterative", random_state=123)
    pd.testing.assert_frame_equal(first, second)

    norm1 = normalize_matrix(first, directions, "robust_minmax")
    norm2 = normalize_matrix(second, directions, "robust_minmax")
    pd.testing.assert_frame_equal(norm1, norm2)

    w1 = compute_weights(norm1, "critic")
    w2 = compute_weights(norm2, "critic")
    pd.testing.assert_series_equal(w1, w2)

    r1 = rank_scores(norm1, w1, "topsis")
    r2 = rank_scores(norm2, w2, "topsis")
    pd.testing.assert_frame_equal(r1, r2)
