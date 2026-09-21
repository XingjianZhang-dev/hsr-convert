from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer


def _all_missing_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].isna().all()]


def impute_matrix(df: pd.DataFrame, method: str, random_state: int = 20260624) -> pd.DataFrame:
    method = method.lower()
    data = df.astype(float).copy()
    all_missing = _all_missing_columns(data)
    if all_missing:
        for col in all_missing:
            data[col] = 0.0

    if method == "median":
        imputer = SimpleImputer(strategy="median")
        values = imputer.fit_transform(data)
    elif method == "knn":
        n_neighbors = min(5, max(1, len(data) - 1))
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        values = imputer.fit_transform(data)
    elif method == "iterative":
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        imputer = IterativeImputer(
            random_state=random_state,
            max_iter=20,
            sample_posterior=False,
            initial_strategy="median",
            skip_complete=True,
        )
        values = imputer.fit_transform(data)
    else:
        raise ValueError(f"Unknown imputation method: {method}")

    out = pd.DataFrame(values, index=df.index, columns=df.columns)
    out = out.replace([np.inf, -np.inf], np.nan)
    if out.isna().any().any():
        out = out.fillna(out.median(numeric_only=True)).fillna(0.0)
    return out
