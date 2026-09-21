from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .conversion_robustness import assign_conversion_profiles, encoded_xy, model_feature_columns, numeric_scaler
from .utils import ROOT


OUTCOME_BASELINE = "cumulative_excess_deaths_per_million_2020_2022"
ALLOWED_PROFILES = [
    "effective converters",
    "capacity under-realizers",
    "adaptive over-performers",
    "structurally vulnerable systems",
    "uncertain / data-limited systems",
]


def potential_with_controls(root: Path = ROOT) -> pd.DataFrame:
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix_expanded.csv")
    controls = matrix[
        [
            c
            for c in [
                "iso3",
                "gdp_per_capita_ppp",
                "age65_share",
                "population_density",
                "population_total",
            ]
            if c in matrix.columns
        ]
    ].drop_duplicates("iso3")
    data = potential.merge(controls, on="iso3", how="left")
    data["log_gdp_per_capita_ppp"] = np.log(data["gdp_per_capita_ppp"].where(data["gdp_per_capita_ppp"] > 0))
    data["population_density_log"] = np.log1p(data["population_density"].clip(lower=0))
    return data


def fit_conversion_for_outcome(
    outcome: pd.DataFrame,
    outcome_col: str,
    variant: str,
    root: Path = ROOT,
    potential_override: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    potential = potential_override.copy() if potential_override is not None else potential_with_controls(root)
    data = potential.merge(outcome[["iso3", outcome_col]], on="iso3", how="inner")
    x, y, work = encoded_xy(data, outcome_col=outcome_col, feature_cols=model_feature_columns())
    diagnostic_cols = ["capacity_score", "preparedness_score", "equity_access_score"]
    missing_diagnostic_cols = [col for col in diagnostic_cols if col not in work.columns and col in data.columns]
    if missing_diagnostic_cols:
        work = work.merge(data[["iso3", *missing_diagnostic_cols]].drop_duplicates("iso3"), on="iso3", how="left")
    if len(work) < 30:
        empty = pd.DataFrame(
            [
                {
                    "evidence_variant": variant,
                    "outcome": outcome_col,
                    "status": "skipped_insufficient_overlap",
                    "n": len(work),
                    "coverage_count": len(work),
                    "missing_count_from_potential_capacity_sample": int(potential["iso3"].nunique() - len(work)),
                    "missing_rate_from_potential_capacity_sample": float(1.0 - len(work) / max(potential["iso3"].nunique(), 1)),
                }
            ]
        )
        return pd.DataFrame(), empty
    model = Pipeline([("scale", numeric_scaler(x)), ("model", Ridge(alpha=10.0))])
    model.fit(x, y)
    pred = model.predict(x)
    conv = work[
        [
            "iso3",
            "country",
            "region",
            "income_group",
            "potential_capacity_score",
            "capacity_score",
            "preparedness_score",
            "vulnerability_score",
            "equity_access_score",
            "method_disagreement_index",
            "expanded_feature_missing_rate",
        ]
    ].copy()
    conv["evidence_variant"] = variant
    conv["outcome"] = outcome_col
    conv["observed_outcome"] = y.to_numpy()
    conv["predicted_outcome"] = pred
    conv["realized_resilience_gap"] = conv["observed_outcome"] - conv["predicted_outcome"]
    std = conv["realized_resilience_gap"].std(ddof=0)
    conv["conversion_efficiency_zscore"] = -(
        conv["realized_resilience_gap"] - conv["realized_resilience_gap"].mean()
    ) / (std if std and np.isfinite(std) else 1.0)
    conv["observed_shock_burden"] = conv["observed_outcome"]
    conv["conversion_profile"] = assign_conversion_profiles(conv)
    block_cols = ["capacity_score", "preparedness_score", "vulnerability_score", "equity_access_score"]
    conv["main_bottleneck_block"] = conv[block_cols].idxmin(axis=1).str.replace("_score", "", regex=False)
    conv["n"] = len(conv)
    conv["coverage_count"] = len(conv)
    conv["missing_count_from_potential_capacity_sample"] = int(potential["iso3"].nunique() - len(conv))
    conv["missing_rate_from_potential_capacity_sample"] = float(1.0 - len(conv) / max(potential["iso3"].nunique(), 1))
    conv["missing_count"] = conv["missing_count_from_potential_capacity_sample"]
    conv["missing_rate"] = conv["missing_rate_from_potential_capacity_sample"]
    diagnostics = pd.DataFrame(
        [
            {
                "evidence_variant": variant,
                "outcome": outcome_col,
                "status": "fit",
                "n": len(conv),
                "coverage_count": len(conv),
                "missing_count_from_potential_capacity_sample": int(potential["iso3"].nunique() - len(conv)),
                "missing_rate_from_potential_capacity_sample": float(1.0 - len(conv) / max(potential["iso3"].nunique(), 1)),
                "model": "ridge",
                "features": ",".join(model_feature_columns()),
            }
        ]
    )
    return conv, diagnostics
