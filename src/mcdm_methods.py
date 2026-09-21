from __future__ import annotations

import numpy as np
import pandas as pd


EPS = 1e-12


def _rank_from_score(score: pd.Series) -> pd.DataFrame:
    score = score.replace([np.inf, -np.inf], np.nan).fillna(score.median())
    rank = score.rank(ascending=False, method="min").astype(int)
    return pd.DataFrame({"score": score, "rank": rank}, index=score.index)


def topsis(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    x = df.to_numpy(dtype=float)
    w = weights.loc[df.columns].to_numpy(dtype=float)
    weighted = x * w
    best = weighted.max(axis=0)
    worst = weighted.min(axis=0)
    d_best = np.sqrt(np.square(weighted - best).sum(axis=1))
    d_worst = np.sqrt(np.square(weighted - worst).sum(axis=1))
    return pd.Series(d_worst / (d_best + d_worst + EPS), index=df.index)


def vikor(df: pd.DataFrame, weights: pd.Series, v: float = 0.5) -> pd.Series:
    x = df.to_numpy(dtype=float)
    w = weights.loc[df.columns].to_numpy(dtype=float)
    best = x.max(axis=0)
    worst = x.min(axis=0)
    gap = np.where(np.isclose(best, worst), 1.0, best - worst)
    regret = w * (best - x) / gap
    s = regret.sum(axis=1)
    r = regret.max(axis=1)
    s_star, s_minus = s.min(), s.max()
    r_star, r_minus = r.min(), r.max()
    q = v * (s - s_star) / (s_minus - s_star + EPS) + (1 - v) * (r - r_star) / (r_minus - r_star + EPS)
    return pd.Series(1.0 - q, index=df.index)


def marcos(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weighted_sum = df.mul(weights.loc[df.columns], axis=1).sum(axis=1)
    lo = weighted_sum.min()
    hi = weighted_sum.max()
    return (weighted_sum - lo) / (hi - lo + EPS)


def edas(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    x = df.to_numpy(dtype=float)
    w = weights.loc[df.columns].to_numpy(dtype=float)
    avg = np.maximum(x.mean(axis=0), EPS)
    pda = np.maximum(0.0, x - avg) / avg
    nda = np.maximum(0.0, avg - x) / avg
    sp = pda @ w
    sn = nda @ w
    nsp = sp / (sp.max() + EPS)
    nsn = 1.0 - sn / (sn.max() + EPS)
    return pd.Series(0.5 * (nsp + nsn), index=df.index)


def codas(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weighted = df.mul(weights.loc[df.columns], axis=1)
    negative = weighted.min(axis=0)
    euclidean = np.sqrt(np.square(weighted - negative).sum(axis=1))
    taxicab = np.abs(weighted - negative).sum(axis=1)
    return pd.Series(euclidean + 0.02 * taxicab, index=df.index)


def waspas(df: pd.DataFrame, weights: pd.Series, lam: float = 0.5) -> pd.Series:
    x = df.clip(lower=EPS)
    w = weights.loc[df.columns]
    additive = x.mul(w, axis=1).sum(axis=1)
    multiplicative = np.exp(np.log(x).mul(w, axis=1).sum(axis=1))
    return lam * additive + (1.0 - lam) * multiplicative


def mabac(df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weighted = df.mul(weights.loc[df.columns], axis=1)
    border = np.exp(np.log(weighted.clip(lower=EPS)).mean(axis=0))
    return (weighted - border).sum(axis=1)


METHODS = {
    "topsis": topsis,
    "vikor": vikor,
    "marcos": marcos,
    "edas": edas,
    "codas": codas,
    "waspas": waspas,
    "mabac": mabac,
}


def rank_scores(df: pd.DataFrame, weights: pd.Series, method: str) -> pd.DataFrame:
    method = method.lower()
    if method not in METHODS:
        raise ValueError(f"Unknown MCDM ranking method: {method}")
    return _rank_from_score(METHODS[method](df, weights))
