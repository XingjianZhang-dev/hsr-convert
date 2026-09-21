from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ROOT, ensure_dirs, save_csv


BLOCKS = ["capacity", "preparedness", "vulnerability", "equity_access"]
BLOCK_SCORE_COLS = [f"{b}_score" for b in BLOCKS]


def _load_profile_scores(root: Path) -> pd.DataFrame:
    profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    scores = pd.read_csv(root / "results/tables/block_scores.csv")
    keep = ["iso3"] + [c for c in scores.columns if c.endswith("_score") or c in ["potential_capacity_score", "full_potential_capacity_score"]]
    merged = profiles.merge(scores[keep], on="iso3", how="left", suffixes=("", "_block"))
    for col in BLOCK_SCORE_COLS:
        alt = f"{col}_block"
        if col in merged.columns and alt in merged.columns:
            merged[col] = merged[col].fillna(merged[alt])
        elif alt in merged.columns:
            merged[col] = merged[alt]
    merged["capacity_quartile"] = pd.qcut(
        merged["potential_capacity_score"].rank(method="first"),
        q=4,
        labels=["Q1_low_capacity", "Q2", "Q3", "Q4_high_capacity"],
    )
    return merged


def _peer_gap_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    peer_specs = [
        ("income_group", "same_income_group"),
        ("region", "same_region"),
        ("capacity_quartile", "same_capacity_quartile"),
    ]
    global_median = df[BLOCK_SCORE_COLS].median()
    for _, row in df.iterrows():
        for block, score_col in zip(BLOCKS, BLOCK_SCORE_COLS):
            for peer_col, peer_label in peer_specs:
                peers = df[df[peer_col].eq(row[peer_col])]
                peer_median = peers[score_col].median() if len(peers) >= 3 else global_median[score_col]
                gap = row[score_col] - peer_median
                rows.append(
                    {
                        "iso3": row["iso3"],
                        "country": row["country"],
                        "conversion_profile": row["conversion_profile"],
                        "peer_group_type": peer_label,
                        "peer_group_value": row[peer_col],
                        "block": block,
                        "country_block_score": row[score_col],
                        "peer_median_block_score": peer_median,
                        "block_gap_country_minus_peer": gap,
                        "negative_gap_is_bottleneck": True,
                        "n_peer_countries": len(peers),
                        "n_validation_countries": len(df),
                        "expanded_feature_missing_rate": row.get("expanded_feature_missing_rate", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def run_conversion_bottlenecks(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    df = _load_profile_scores(root)
    gap = _peer_gap_rows(df)
    save_csv(gap, root / "results/tables/block_gap_analysis.csv")

    income_gap = gap[gap["peer_group_type"].eq("same_income_group")].copy()
    pivot = income_gap.pivot_table(index=["iso3", "country", "conversion_profile"], columns="block", values="block_gap_country_minus_peer").reset_index()
    for block in BLOCKS:
        if block not in pivot.columns:
            pivot[block] = np.nan
    pivot["weakest_bottleneck_block"] = pivot[BLOCKS].idxmin(axis=1)
    pivot["weakest_bottleneck_gap"] = pivot[BLOCKS].min(axis=1)
    pivot["second_bottleneck_block"] = pivot[BLOCKS].apply(lambda row: row.sort_values().index[1] if row.notna().sum() > 1 else np.nan, axis=1)
    conv = pd.read_csv(root / "results/tables/conversion_efficiency_scores.csv")
    bottleneck = pivot.merge(
        conv[["iso3", "conversion_efficiency_zscore", "realized_resilience_gap", "method_disagreement_index", "expanded_feature_missing_rate"]],
        on="iso3",
        how="left",
    )
    bottleneck["n_validation_countries"] = len(df)
    bottleneck["missing_count_from_profile_sample"] = int(df["iso3"].nunique() - bottleneck["iso3"].nunique())
    bottleneck["missing_rate_from_profile_sample"] = float(1.0 - bottleneck["iso3"].nunique() / max(df["iso3"].nunique(), 1))
    save_csv(bottleneck.sort_values("conversion_efficiency_zscore"), root / "results/tables/conversion_bottleneck_scores.csv")

    under = bottleneck[bottleneck["conversion_profile"].eq("capacity under-realizers")].copy()
    if under.empty:
        under = bottleneck.nsmallest(12, "conversion_efficiency_zscore").copy()
        under["diagnostic_scope"] = "lowest_conversion_efficiency_no_explicit_under_realizer_profile"
    else:
        under["diagnostic_scope"] = "capacity_under_realizers"
    under["diagnostic_note"] = "Conversion bottleneck diagnostic only; no causal policy prescription."
    save_csv(under.sort_values("conversion_efficiency_zscore"), root / "results/tables/under_realizer_diagnostics.csv")

    _plot_bottleneck_heatmap(bottleneck, root)
    _plot_profile_radar(df, root)
    return bottleneck, gap


def _plot_bottleneck_heatmap(bottleneck: pd.DataFrame, root: Path) -> None:
    plot = bottleneck.sort_values("conversion_efficiency_zscore").head(40).copy()
    data = plot[BLOCKS].to_numpy(dtype=float)
    plt.figure(figsize=(9, max(6, 0.18 * len(plot))))
    im = plt.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.35, vmax=0.35)
    plt.yticks(range(len(plot)), plot["country"], fontsize=7)
    plt.xticks(range(len(BLOCKS)), BLOCKS, rotation=30, ha="right")
    plt.colorbar(im, label="Block gap vs same-income median")
    plt.title("Conversion bottleneck heatmap")
    plt.tight_layout()
    plt.savefig(root / "results/figures/conversion_bottleneck_heatmap.pdf", bbox_inches="tight")
    plt.close()


def _plot_profile_radar(df: pd.DataFrame, root: Path) -> None:
    profile_means = df.groupby("conversion_profile")[BLOCK_SCORE_COLS].mean()
    labels = [b.replace("_", " ") for b in BLOCKS]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    for profile, row in profile_means.iterrows():
        values = [row[col] for col in BLOCK_SCORE_COLS]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, label=profile)
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Profile block signatures")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    plt.tight_layout()
    plt.savefig(root / "results/figures/profile_block_signature_radar.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_conversion_bottlenecks()
