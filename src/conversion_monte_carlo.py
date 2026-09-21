from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .conversion_robustness import PROFILE_LABELS, assign_conversion_profiles, encoded_xy, load_conversion_model_data, model_feature_columns, numeric_scaler
from .utils import ROOT, ensure_dirs, save_csv


BLOCKS = ["capacity", "preparedness", "vulnerability", "equity_access"]
BLOCK_SCORE_COLS = [f"{b}_score" for b in BLOCKS]
OUTCOME = "cumulative_excess_deaths_per_million_2020_2022"


def _simulation_dataset(root: Path) -> pd.DataFrame:
    data = load_conversion_model_data(root)
    data = data.dropna(subset=[OUTCOME] + model_feature_columns()).copy()
    return data.reset_index(drop=True)


def _profile_probabilities(profile_draws: pd.DataFrame, countries: pd.DataFrame, n_sim: int, root: Path) -> pd.DataFrame:
    rows = []
    for _, row in countries.iterrows():
        probs = profile_draws[row["iso3"]].value_counts(normalize=True).to_dict()
        for label in PROFILE_LABELS:
            rows.append(
                {
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "profile": label,
                    "profile_probability": float(probs.get(label, 0.0)),
                    "n_simulations": n_sim,
                    "coverage_count": len(countries),
                    "missing_count_from_potential_capacity_sample": int(
                        pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique() - len(countries)
                    ),
                    "missing_rate_from_potential_capacity_sample": float(
                        1.0
                        - len(countries)
                        / max(pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique(), 1)
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_conversion_monte_carlo(
    root: Path = ROOT,
    n_sim: int = 1000,
    block_noise_sd: float = 0.025,
    noise_multiplier: float = 1.0,
    seed: int = 20260624,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monte Carlo perturbation of block weights, block scores and the ridge fit.

    ``noise_multiplier`` scales both the block-score noise (default sd 0.025) and the method-disagreement
    noise; it is used by the revision sensitivity analysis. ``save=False`` returns the summaries without
    overwriting the baseline tables and figures.
    """

    ensure_dirs(root)
    data = _simulation_dataset(root)
    rng = np.random.default_rng(seed)
    eff = np.full((n_sim, len(data)), np.nan)
    profiles: list[list[str]] = []
    bottlenecks: list[list[str]] = []
    block_base = data[BLOCK_SCORE_COLS].to_numpy(dtype=float)
    method_noise_scale = noise_multiplier * (0.015 + 0.08 * data["method_disagreement_index"].to_numpy(dtype=float))
    block_noise_sd = block_noise_sd * noise_multiplier

    for sim in range(n_sim):
        weights = rng.dirichlet(np.ones(len(BLOCKS)))
        noisy_blocks = np.clip(block_base + rng.normal(0, block_noise_sd, size=block_base.shape), 0, 1)
        if rng.random() < 0.15:
            drop = rng.integers(0, len(BLOCKS))
            weights[drop] = 0
            weights = weights / weights.sum()
        potential = np.clip(noisy_blocks @ weights + rng.normal(0, method_noise_scale), 0, 1)
        sim_data = data.copy()
        sim_data["potential_capacity_score"] = potential
        x, y, work = encoded_xy(sim_data, outcome_col=OUTCOME, feature_cols=model_feature_columns())
        idx = rng.integers(0, len(work), len(work))
        model = Pipeline([("scale", numeric_scaler(x)), ("model", Ridge(alpha=10.0))])
        model.fit(x.iloc[idx], y.iloc[idx])
        pred = model.predict(x)
        frame = work[[
            "iso3",
            "country",
            "region",
            "income_group",
            "potential_capacity_score",
            "method_disagreement_index",
            "expanded_feature_missing_rate",
        ]].copy()
        frame["observed_shock_burden"] = y.to_numpy()
        frame["predicted_shock_burden"] = pred
        frame["realized_resilience_gap"] = frame["observed_shock_burden"] - frame["predicted_shock_burden"]
        frame["conversion_efficiency_zscore"] = -(
            frame["realized_resilience_gap"] - frame["realized_resilience_gap"].mean()
        ) / frame["realized_resilience_gap"].std(ddof=0)
        frame["conversion_profile"] = assign_conversion_profiles(frame)
        eff[sim, :] = frame["conversion_efficiency_zscore"].to_numpy()
        profiles.append(frame["conversion_profile"].tolist())
        gaps = pd.DataFrame(noisy_blocks, columns=BLOCKS)
        bottlenecks.append(gaps.idxmin(axis=1).tolist())

    countries = data[["iso3", "country"]].copy()
    profile_draws = pd.DataFrame(profiles, columns=data["iso3"])
    bottleneck_draws = pd.DataFrame(bottlenecks, columns=data["iso3"])
    rows = []
    for j, (_, row) in enumerate(countries.iterrows()):
        vals = eff[:, j]
        probs = profile_draws[row["iso3"]].value_counts(normalize=True).to_dict()
        rows.append(
            {
                "iso3": row["iso3"],
                "country": row["country"],
                "n_simulations": n_sim,
                "coverage_count": len(countries),
                "missing_count_from_potential_capacity_sample": int(
                    pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique() - len(countries)
                ),
                "missing_rate_from_potential_capacity_sample": float(
                    1.0
                    - len(countries)
                    / max(pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique(), 1)
                ),
                "conversion_efficiency_mean": float(np.nanmean(vals)),
                "conversion_efficiency_ci_low": float(np.nanpercentile(vals, 2.5)),
                "conversion_efficiency_ci_high": float(np.nanpercentile(vals, 97.5)),
                "conversion_efficiency_std": float(np.nanstd(vals)),
                "most_likely_profile": max(probs, key=probs.get),
                "profile_stability_score": float(max(probs.values())),
                "robustness_category": "robust" if max(probs.values()) >= 0.75 else "unstable",
            }
        )
    summary = pd.DataFrame(rows)
    profile_probs = _profile_probabilities(profile_draws, countries, n_sim, root)
    bottleneck_rows = []
    for _, row in countries.iterrows():
        probs = bottleneck_draws[row["iso3"]].value_counts(normalize=True).to_dict()
        for block in BLOCKS:
            bottleneck_rows.append(
                {
                    "iso3": row["iso3"],
                    "country": row["country"],
                    "bottleneck_block": block,
                    "bottleneck_probability": float(probs.get(block, 0.0)),
                    "n_simulations": n_sim,
                    "coverage_count": len(countries),
                    "missing_count": 0,
                    "missing_rate": 0.0,
                }
            )
    bottleneck = pd.DataFrame(bottleneck_rows)
    if not save:
        return summary, profile_probs
    save_csv(summary, root / "results/tables/conversion_monte_carlo_summary.csv")
    save_csv(profile_probs, root / "results/tables/profile_probability_table.csv")
    save_csv(bottleneck, root / "results/tables/bottleneck_stability_table.csv")
    _plot_mc_intervals(summary, root / "results/figures/conversion_monte_carlo_intervals.pdf")
    _plot_profile_probability(profile_probs, root / "results/figures/profile_probability_heatmap.pdf")
    _plot_bottleneck_probability(bottleneck, root / "results/figures/bottleneck_stability_heatmap.pdf")
    return summary, profile_probs


def _plot_mc_intervals(summary: pd.DataFrame, path: Path) -> None:
    plot = pd.concat([summary.nsmallest(20, "conversion_efficiency_mean"), summary.nlargest(20, "conversion_efficiency_mean")])
    plot = plot.sort_values("conversion_efficiency_mean")
    y = np.arange(len(plot))
    plt.figure(figsize=(9, max(7, 0.18 * len(plot))))
    plt.hlines(y, plot["conversion_efficiency_ci_low"], plot["conversion_efficiency_ci_high"], color="#bcbddc")
    plt.scatter(plot["conversion_efficiency_mean"], y, color="#54278f", s=14)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.yticks(y, plot["country"], fontsize=7)
    plt.xlabel("Monte Carlo conversion efficiency")
    plt.title("Conversion Monte Carlo intervals")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_profile_probability(profile_probs: pd.DataFrame, path: Path) -> None:
    pivot = profile_probs.pivot_table(index="country", columns="profile", values="profile_probability")
    order = pivot.max(axis=1).sort_values().head(60).index
    plot = pivot.loc[order, PROFILE_LABELS]
    plt.figure(figsize=(11, 9))
    plt.imshow(plot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.yticks(range(len(plot.index)), plot.index, fontsize=7)
    plt.xticks(range(len(PROFILE_LABELS)), PROFILE_LABELS, rotation=45, ha="right", fontsize=8)
    plt.colorbar(label="Profile probability")
    plt.title("Least certain profile probabilities")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_bottleneck_probability(bottleneck: pd.DataFrame, path: Path) -> None:
    pivot = bottleneck.pivot_table(index="country", columns="bottleneck_block", values="bottleneck_probability")
    order = pivot.max(axis=1).sort_values().head(60).index
    plot = pivot.loc[order, BLOCKS]
    plt.figure(figsize=(9, 9))
    plt.imshow(plot.values, aspect="auto", vmin=0, vmax=1, cmap="magma")
    plt.yticks(range(len(plot.index)), plot.index, fontsize=7)
    plt.xticks(range(len(BLOCKS)), BLOCKS, rotation=30, ha="right")
    plt.colorbar(label="Bottleneck probability")
    plt.title("Bottleneck stability")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_conversion_monte_carlo()
