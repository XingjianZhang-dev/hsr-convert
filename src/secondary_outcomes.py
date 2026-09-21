from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .conversion_robustness import assign_conversion_profiles, encoded_xy, load_conversion_model_data, model_feature_columns, numeric_scaler
from .utils import ROOT, ensure_dirs, save_csv


RAW_DIRS = [
    "data/raw/who_pulse",
    "data/raw/life_expectancy",
    "data/raw/secondary_outcomes",
]


README_TEXT = """# Secondary Outcome Manual Data Folder

Place legitimate country-level secondary outcome CSV files here. The pipeline will not fabricate values.

Required columns:
- `iso3`: ISO-3 country code.
- one or more numeric outcome columns.

Recommended metadata columns:
- `country`
- `source`
- `years`
- `direction`, with `higher_worse` or `higher_better`

Shock-period or post-shock outcomes are validation-only and must never be used as pre-shock features.
"""


def _ensure_secondary_dirs(root: Path) -> None:
    for rel in RAW_DIRS:
        path = root / rel
        path.mkdir(parents=True, exist_ok=True)
        readme = path / "README.md"
        if not readme.exists():
            readme.write_text(README_TEXT, encoding="utf-8")


def _available_outcomes(root: Path) -> list[dict[str, object]]:
    rows = []
    for rel in RAW_DIRS:
        for path in sorted((root / rel).glob("*.csv")):
            df = pd.read_csv(path)
            if "iso3" not in df.columns:
                continue
            numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in {"year"}]
            for col in numeric:
                rows.append(
                    {
                        "path": path,
                        "source": rel,
                        "outcome": col,
                        "data": df[["iso3", col] + ([c for c in ["country", "years", "direction"] if c in df.columns])].copy(),
                    }
                )
    return rows


def _write_skip_outputs(root: Path, reason: str) -> None:
    inventory = pd.DataFrame(
        [
            {
                "status": "skipped_no_legitimate_country_level_file",
                "source": "",
                "outcome": "",
                "years": "",
                "higher_is": "",
                "n_countries": 0,
                "coverage_count": 0,
                "missing_count_from_scoring_sample": pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique(),
                "missing_rate_from_scoring_sample": 1.0,
                "overlap_with_owid_excess_mortality_sample": 0,
                "note": reason,
            }
        ]
    )
    save_csv(inventory, root / "results/tables/secondary_outcome_inventory.csv")
    for output in [
        "secondary_outcome_validation.csv",
        "profile_overlap_across_outcomes.csv",
        "under_realizer_overlap_across_outcomes.csv",
    ]:
        save_csv(inventory, root / f"results/tables/{output}")
    _placeholder(root / "results/figures/secondary_outcome_validation_heatmap.pdf", "No secondary outcome file available")
    _placeholder(root / "results/figures/profile_overlap_across_outcomes.pdf", "No secondary outcome file available")


def run_secondary_outcome_validation(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    _ensure_secondary_dirs(root)
    outcomes = _available_outcomes(root)
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    owid = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    if not outcomes:
        _write_skip_outputs(root, "No CSV with `iso3` and numeric country-level secondary outcome was found.")
        return pd.read_csv(root / "results/tables/secondary_outcome_inventory.csv"), pd.DataFrame()

    inventory_rows = []
    validation_rows = []
    profile_frames = []
    under_frames = []
    for item in outcomes:
        df = item["data"].copy()
        df["iso3"] = df["iso3"].astype(str).str.upper()
        outcome = str(item["outcome"])
        direction = "higher_worse"
        if "direction" in df.columns and df["direction"].notna().any():
            direction = str(df["direction"].dropna().iloc[0])
        data = load_conversion_model_data(root).drop(columns=["cumulative_excess_deaths_per_million_2020_2022"])
        merged = potential[["iso3"]].merge(df[["iso3", outcome]], on="iso3", how="left")
        overlap_scoring = int(merged[outcome].notna().sum())
        overlap_owid = int(owid[["iso3"]].merge(df[["iso3", outcome]], on="iso3", how="inner")[outcome].notna().sum())
        inventory_rows.append(
            {
                "status": "loaded",
                "source": item["source"],
                "file_name": str(Path(item["path"]).relative_to(root)),
                "outcome": outcome,
                "years": str(df["years"].dropna().iloc[0]) if "years" in df.columns and df["years"].notna().any() else "not recorded",
                "higher_is": direction,
                "n_countries": df["iso3"].nunique(),
                "coverage_count": overlap_scoring,
                "missing_count_from_scoring_sample": int(potential["iso3"].nunique() - overlap_scoring),
                "missing_rate_from_scoring_sample": float(1.0 - overlap_scoring / max(potential["iso3"].nunique(), 1)),
                "overlap_with_owid_excess_mortality_sample": overlap_owid,
                "note": "Validation-only secondary outcome.",
            }
        )
        model_data = data.merge(df[["iso3", outcome]], on="iso3", how="inner")
        if direction == "higher_better":
            model_data[outcome] = -model_data[outcome]
        x, y, work = encoded_xy(model_data, outcome_col=outcome, feature_cols=model_feature_columns())
        if len(work) < 30:
            validation_rows.append(
                {
                    "outcome": outcome,
                    "status": "skipped_insufficient_overlap",
                    "n": len(work),
                    "coverage_count": len(work),
                    "missing_count": int(potential["iso3"].nunique() - len(work)),
                    "missing_rate": float(1.0 - len(work) / max(potential["iso3"].nunique(), 1)),
                }
            )
            continue
        model = Pipeline([("scale", numeric_scaler(x)), ("model", Ridge(alpha=10.0))])
        model.fit(x, y)
        pred = model.predict(x)
        conv = work[["iso3", "country", "region", "income_group", "potential_capacity_score", "method_disagreement_index", "expanded_feature_missing_rate"]].copy()
        conv["observed_shock_burden"] = y.to_numpy()
        conv["predicted_shock_burden"] = pred
        conv["realized_resilience_gap"] = conv["observed_shock_burden"] - conv["predicted_shock_burden"]
        conv["conversion_efficiency_zscore"] = -(
            conv["realized_resilience_gap"] - conv["realized_resilience_gap"].mean()
        ) / conv["realized_resilience_gap"].std(ddof=0)
        conv["conversion_profile"] = assign_conversion_profiles(conv)
        validation_rows.append(
            {
                "outcome": outcome,
                "status": "fit",
                "n": len(conv),
                "coverage_count": len(conv),
                "missing_count": int(potential["iso3"].nunique() - len(conv)),
                "missing_rate": float(1.0 - len(conv) / max(potential["iso3"].nunique(), 1)),
                "spearman_efficiency_vs_outcome": spearmanr(conv["conversion_efficiency_zscore"], conv["observed_shock_burden"]).statistic,
                "profile_counts": str(conv["conversion_profile"].value_counts().to_dict()),
                "top_under_realizers": ",".join(conv.nsmallest(10, "conversion_efficiency_zscore")["iso3"]),
                "top_over_performers": ",".join(conv.nlargest(10, "conversion_efficiency_zscore")["iso3"]),
            }
        )
        conv["outcome"] = outcome
        profile_frames.append(conv[["outcome", "iso3", "conversion_profile", "conversion_efficiency_zscore"]])
        under_frames.append(conv.sort_values("conversion_efficiency_zscore").head(20)[["outcome", "iso3", "conversion_efficiency_zscore"]])

    inventory = pd.DataFrame(inventory_rows)
    validation = pd.DataFrame(validation_rows)
    save_csv(inventory, root / "results/tables/secondary_outcome_inventory.csv")
    save_csv(validation, root / "results/tables/secondary_outcome_validation.csv")
    _cross_outcome_tables(profile_frames, under_frames, root)
    _plot_secondary_validation(validation, root)
    return inventory, validation


def _cross_outcome_tables(profile_frames: list[pd.DataFrame], under_frames: list[pd.DataFrame], root: Path) -> None:
    if len(profile_frames) < 2:
        single = pd.DataFrame(
            [
                {
                    "status": "skipped_insufficient_number_of_secondary_outcomes",
                    "n": len(profile_frames[0]) if profile_frames else 0,
                    "coverage_count": len(profile_frames[0]) if profile_frames else 0,
                    "missing_count": 0,
                    "missing_rate": 0.0,
                }
            ]
        )
        save_csv(single, root / "results/tables/profile_overlap_across_outcomes.csv")
        save_csv(single, root / "results/tables/under_realizer_overlap_across_outcomes.csv")
        _placeholder(root / "results/figures/profile_overlap_across_outcomes.pdf", "Insufficient secondary outcomes")
        return
    profiles = pd.concat(profile_frames, ignore_index=True)
    rows = []
    under_rows = []
    for a, b in combinations(sorted(profiles["outcome"].unique()), 2):
        left = profiles[profiles["outcome"].eq(a)]
        right = profiles[profiles["outcome"].eq(b)]
        merged = left.merge(right, on="iso3", suffixes=("_a", "_b"))
        rows.append(
            {
                "outcome_a": a,
                "outcome_b": b,
                "n": len(merged),
                "coverage_count": len(merged),
                "missing_count": 0,
                "missing_rate": 0.0,
                "efficiency_spearman": spearmanr(
                    merged["conversion_efficiency_zscore_a"], merged["conversion_efficiency_zscore_b"]
                ).statistic,
                "profile_overlap": float((merged["conversion_profile_a"] == merged["conversion_profile_b"]).mean()),
            }
        )
        ua = set(left.nsmallest(10, "conversion_efficiency_zscore")["iso3"])
        ub = set(right.nsmallest(10, "conversion_efficiency_zscore")["iso3"])
        under_rows.append(
            {
                "outcome_a": a,
                "outcome_b": b,
                "n": len(merged),
                "coverage_count": len(merged),
                "missing_count": 0,
                "missing_rate": 0.0,
                "top10_under_realizer_overlap": len(ua & ub) / 10,
            }
        )
    overlap = pd.DataFrame(rows)
    under = pd.DataFrame(under_rows)
    save_csv(overlap, root / "results/tables/profile_overlap_across_outcomes.csv")
    save_csv(under, root / "results/tables/under_realizer_overlap_across_outcomes.csv")
    plt.figure(figsize=(7, 4))
    plt.barh(overlap["outcome_a"] + " vs " + overlap["outcome_b"], overlap["profile_overlap"])
    plt.xlim(0, 1)
    plt.xlabel("Profile overlap")
    plt.title("Profile overlap across secondary outcomes")
    plt.tight_layout()
    plt.savefig(root / "results/figures/profile_overlap_across_outcomes.pdf", bbox_inches="tight")
    plt.close()


def _plot_secondary_validation(validation: pd.DataFrame, root: Path) -> None:
    if "spearman_efficiency_vs_outcome" not in validation.columns:
        _placeholder(root / "results/figures/secondary_outcome_validation_heatmap.pdf", "No fitted secondary outcomes")
        return
    plot = validation.dropna(subset=["spearman_efficiency_vs_outcome"])
    if plot.empty:
        _placeholder(root / "results/figures/secondary_outcome_validation_heatmap.pdf", "No fitted secondary outcomes")
        return
    plt.figure(figsize=(7, max(3, 0.4 * len(plot))))
    plt.imshow(plot[["spearman_efficiency_vs_outcome"]].to_numpy(), aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    plt.yticks(range(len(plot)), plot["outcome"])
    plt.xticks([0], ["Efficiency vs outcome"])
    plt.colorbar(label="Spearman r")
    plt.title("Secondary outcome validation")
    plt.tight_layout()
    plt.savefig(root / "results/figures/secondary_outcome_validation_heatmap.pdf", bbox_inches="tight")
    plt.close()


def _placeholder(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_secondary_outcome_validation()
