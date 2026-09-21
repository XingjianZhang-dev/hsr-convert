from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ROOT


def _savefig(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def plot_missingness_heatmap(matrix: pd.DataFrame, feature_cols: list[str], path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/missingness_heatmap.pdf"
    data = matrix.set_index("iso3")[feature_cols].isna().astype(int)
    data = data.loc[data.sum(axis=1).sort_values(ascending=False).index]
    fig_h = max(6, min(18, 0.04 * len(data) + 3))
    plt.figure(figsize=(11, fig_h))
    plt.imshow(data.values, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks([])
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=70, ha="right", fontsize=8)
    plt.title("Missingness in pre-shock feature matrix")
    plt.xlabel("Indicator")
    plt.ylabel("Countries sorted by missing feature count")
    cbar = plt.colorbar(shrink=0.7)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["observed", "missing"])
    return _savefig(path)


def plot_correlation_heatmap(matrix: pd.DataFrame, feature_cols: list[str], path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/correlation_heatmap.pdf"
    corr = matrix[feature_cols].corr(method="spearman", min_periods=20)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(feature_cols)), feature_cols, rotation=70, ha="right", fontsize=8)
    plt.yticks(range(len(feature_cols)), feature_cols, fontsize=8)
    plt.title("Spearman correlation among pre-shock indicators")
    plt.colorbar(im, shrink=0.8)
    return _savefig(path)


def plot_baseline_rankings(baseline: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/baseline_top_bottom_ranks.pdf"
    preferred = baseline[baseline["ranking_method"] == baseline["ranking_method"].iloc[0]].copy()
    top = preferred.nsmallest(10, "rank")
    bottom = preferred.nlargest(10, "rank").sort_values("rank")
    plot_df = pd.concat([top, bottom])
    colors = ["#2b8cbe" if r <= 10 else "#d95f0e" for r in plot_df["rank"]]
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["country"], plot_df["score"], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Baseline MCDM score")
    plt.title("Baseline top and bottom countries")
    return _savefig(path)


def plot_rank_intervals(consensus: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/rank_interval_plot.pdf"
    plot_df = consensus.sort_values("median_rank").copy()
    max_rows = min(80, len(plot_df))
    plot_df = pd.concat([plot_df.head(max_rows // 2), plot_df.tail(max_rows // 2)])
    y = np.arange(len(plot_df))
    plt.figure(figsize=(10, max(7, 0.16 * len(plot_df))))
    plt.hlines(y, plot_df["rank_p05"], plot_df["rank_p95"], color="#8da0cb", linewidth=2)
    plt.scatter(plot_df["median_rank"], y, color="#1b1b1b", s=14, zorder=3)
    plt.yticks(y, plot_df["country"], fontsize=7)
    plt.gca().invert_yaxis()
    plt.xlabel("Rank across methodological pipelines")
    plt.title("Consensus rank intervals, 5th to 95th percentile")
    return _savefig(path)


def plot_top_quartile_probability(consensus: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/top_quartile_probability_heatmap.pdf"
    plot_df = consensus.sort_values("top_quartile_probability", ascending=False).head(60)
    plt.figure(figsize=(9, 8))
    plt.imshow(plot_df[["top_quartile_probability"]].values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.yticks(range(len(plot_df)), plot_df["country"], fontsize=7)
    plt.xticks([0], ["Top-quartile probability"])
    plt.title("Countries most often in the top resilience quartile")
    plt.colorbar(shrink=0.8)
    return _savefig(path)


def plot_method_disagreement(consensus: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/method_disagreement_plot.pdf"
    plot_df = consensus.sort_values("method_disagreement_index", ascending=False).head(40)
    plt.figure(figsize=(9, 7))
    plt.barh(plot_df["country"], plot_df["method_disagreement_index"], color="#756bb1")
    plt.gca().invert_yaxis()
    plt.xlabel("Method disagreement index")
    plt.title("Most method-sensitive country ranks")
    return _savefig(path)


def plot_resilience_vs_outcome(validation_df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/resilience_vs_excess_mortality.pdf"
    x = validation_df["resilience_score_rank_based"].astype(float)
    y = validation_df["cumulative_excess_deaths_per_million_2020_2022"].astype(float)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=25, alpha=0.75, color="#2b8cbe", edgecolor="white", linewidth=0.3)
    if len(validation_df) >= 3:
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(x.min(), x.max(), 100)
        plt.plot(xs, coef[0] * xs + coef[1], color="#d95f0e", linewidth=1.5)
    plt.xlabel("Pre-shock resilience score, rank-based")
    plt.ylabel("Cumulative excess deaths per million, 2020-2022")
    plt.title("Pre-shock resilience versus later excess mortality")
    return _savefig(path)


def plot_leave_region_out(leave_region: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or ROOT / "results/figures/leave_region_out_validation.pdf"
    if leave_region.empty:
        plt.figure(figsize=(7, 3))
        plt.text(0.5, 0.5, "Insufficient regional coverage", ha="center", va="center")
        plt.axis("off")
        return _savefig(path)
    plot_df = leave_region.sort_values("spearman_r")
    plt.figure(figsize=(8, max(4, 0.4 * len(plot_df))))
    plt.barh(plot_df["held_out_region"], plot_df["spearman_r"], color="#31a354")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Spearman r after excluding region")
    plt.title("Leave-one-region-out validation sensitivity")
    return _savefig(path)

