from __future__ import annotations

import numpy as np
import pandas as pd


def _minmax(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if not np.isfinite(min_val) or not np.isfinite(max_val) or np.isclose(max_val, min_val):
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def _orient(series: pd.Series, direction: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    if direction == "cost":
        return -x
    if direction == "beneficial":
        return x
    raise ValueError(f"Unknown indicator direction: {direction}")


def normalize_matrix(df: pd.DataFrame, directions: dict[str, str], method: str) -> pd.DataFrame:
    method = method.lower()
    oriented = pd.DataFrame(index=df.index)
    for col in df.columns:
        oriented[col] = _orient(df[col], directions[col])

    if method == "minmax":
        out = oriented.apply(_minmax, axis=0)
    elif method == "robust_minmax":
        clipped = pd.DataFrame(index=oriented.index)
        for col in oriented.columns:
            lo = oriented[col].quantile(0.05)
            hi = oriented[col].quantile(0.95)
            clipped[col] = oriented[col].clip(lo, hi)
        out = clipped.apply(_minmax, axis=0)
    elif method == "vector":
        scaled = oriented.apply(_minmax, axis=0)
        out = pd.DataFrame(index=df.index)
        for col in scaled.columns:
            denom = float(np.sqrt(np.square(scaled[col]).sum()))
            out[col] = scaled[col] / denom if denom > 0 else 0.5
    elif method == "zscore_minmax":
        z = pd.DataFrame(index=oriented.index)
        for col in oriented.columns:
            std = oriented[col].std(ddof=0)
            z[col] = (oriented[col] - oriented[col].mean()) / std if std and np.isfinite(std) else 0.0
        out = z.apply(_minmax, axis=0)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return out.clip(lower=0.0).fillna(0.5)
