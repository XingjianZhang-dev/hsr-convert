from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, ElasticNetCV, HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.conversion_robustness import (
    OUTCOME,
    assign_conversion_profiles,
    encoded_xy,
    load_conversion_model_data,
    model_feature_columns,
    numeric_scaler,
)
from src.utils import ROOT

SEED = 20260921
RESULTS_DIR = Path(os.environ.get("HSR_REVISION_RESULTS", ROOT / "results" / "revision"))
FIG_DIR = RESULTS_DIR / "figures"
NUMBERS_PATH = RESULTS_DIR / "revision_numbers.json"

PROFILE_ORDER = [
    "effective converters",
    "capacity under-realizers",
    "adaptive over-performers",
    "structurally vulnerable systems",
    "uncertain / data-limited systems",
]


def ensure_results_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def model_data(root: Path = ROOT) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """The 106-country ridge training set, encoded exactly as in the package."""

    data = load_conversion_model_data(root)
    x, y, work = encoded_xy(data)
    work = work.rename(columns={OUTCOME: "observed_shock_burden"}).reset_index(drop=True)
    return x.reset_index(drop=True), y.reset_index(drop=True), work


def baseline_profiles(root: Path = ROOT) -> pd.DataFrame:
    return pd.read_csv(root / "results/tables/conversion_profiles.csv")


def make_model(name: str, x: pd.DataFrame, seed: int = SEED):
    """Candidate expected-burden models. Linear/ridge/Huber/RF match the package; ENet and HGB are new."""

    scaler = numeric_scaler(x)
    if name == "intercept_only":
        from sklearn.dummy import DummyRegressor

        return DummyRegressor(strategy="mean")
    if name == "linear":
        return Pipeline([("scale", scaler), ("model", LinearRegression())])
    if name == "ridge":
        return Pipeline([("scale", scaler), ("model", Ridge(alpha=10.0))])
    if name == "huber":
        return Pipeline([("scale", scaler), ("model", HuberRegressor(max_iter=1000))])
    if name == "elastic_net":
        # Hyper-parameters are chosen by an inner 5-fold CV on the training fold only (nested CV).
        return Pipeline(
            [
                ("scale", scaler),
                ("model", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=50, cv=5, max_iter=20000, random_state=seed)),
            ]
        )
    if name == "random_forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=4, random_state=20260624)
    if name == "gradient_boosting":
        # Small-sample settings: shallow trees, slow learning rate, early stopping off for determinism.
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=5, l2_regularization=1.0, random_state=seed
        )
    raise ValueError(name)


def update_numbers(section: str, payload: dict[str, Any]) -> None:
    ensure_results_dirs()
    numbers: dict[str, Any] = {}
    if NUMBERS_PATH.exists():
        numbers = json.loads(NUMBERS_PATH.read_text(encoding="utf-8"))
    numbers[section] = _jsonable(payload)
    NUMBERS_PATH.write_text(json.dumps(numbers, indent=2, sort_keys=True), encoding="utf-8")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    return obj


def efficiency_from_predictions(work: pd.DataFrame, predicted: np.ndarray) -> pd.DataFrame:
    """Residual conversion + profile rule, identical to the package's baseline path."""

    out = work.copy()
    out["predicted_shock_burden"] = predicted
    out["realized_resilience_gap"] = out["observed_shock_burden"] - out["predicted_shock_burden"]
    std = out["realized_resilience_gap"].std(ddof=0)
    out["conversion_efficiency_zscore"] = -(out["realized_resilience_gap"] - out["realized_resilience_gap"].mean()) / (
        std if std and np.isfinite(std) else 1.0
    )
    out["conversion_profile"] = assign_conversion_profiles(out)
    return out


__all__ = [
    "SEED",
    "RESULTS_DIR",
    "FIG_DIR",
    "PROFILE_ORDER",
    "ensure_results_dirs",
    "model_data",
    "baseline_profiles",
    "make_model",
    "update_numbers",
    "efficiency_from_predictions",
    "model_feature_columns",
    "assign_conversion_profiles",
    "numeric_scaler",
    "OUTCOME",
    "ROOT",
    "ColumnTransformer",
    "StandardScaler",
    "ElasticNet",
]
