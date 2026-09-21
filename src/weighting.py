from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def _safe_weights(raw: np.ndarray, n: int) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    raw = np.maximum(raw, 0.0)
    total = raw.sum()
    if total <= EPS:
        return np.repeat(1.0 / n, n)
    return raw / total


def compute_weights(df: pd.DataFrame, method: str) -> pd.Series:
    method = method.lower()
    x = df.astype(float).to_numpy()
    n_criteria = x.shape[1]

    if method == "equal":
        weights = np.repeat(1.0 / n_criteria, n_criteria)
    elif method == "std":
        weights = _safe_weights(np.std(x, axis=0, ddof=0), n_criteria)
    elif method == "entropy":
        col_sums = x.sum(axis=0)
        p = np.divide(x, col_sums + EPS)
        entropy = -(p * np.log(p + EPS)).sum(axis=0) / np.log(max(len(df), 2))
        weights = _safe_weights(1.0 - entropy, n_criteria)
    elif method == "critic":
        std = np.std(x, axis=0, ddof=0)
        corr = np.corrcoef(x, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        conflict = np.sum(1.0 - corr, axis=0)
        weights = _safe_weights(std * conflict, n_criteria)
    elif method == "merec":
        base = x.mean(axis=1)
        impacts = []
        for j in range(n_criteria):
            reduced = np.delete(x, j, axis=1)
            reduced_mean = reduced.mean(axis=1) if reduced.shape[1] else np.zeros(len(df))
            impacts.append(np.abs(base - reduced_mean).sum())
        weights = _safe_weights(np.asarray(impacts), n_criteria)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return pd.Series(weights, index=df.columns, name=method)
