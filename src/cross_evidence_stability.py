from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stage4_common import ALLOWED_PROFILES
from .utils import ROOT, ensure_dirs, save_csv


def _baseline_variant(root: Path) -> pd.DataFrame:
    profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    bottleneck = pd.read_csv(root / "results/tables/conversion_bottleneck_scores.csv")[
        ["iso3", "weakest_bottleneck_block"]
    ].rename(columns={"weakest_bottleneck_block": "main_bottleneck_block"})
    out = profiles[
        [
            "iso3",
            "country",
            "conversion_profile",
            "conversion_efficiency_zscore",
            "realized_resilience_gap",
        ]
    ].merge(bottleneck, on="iso3", how="left")
    out["evidence_variant"] = "baseline_owid_excess_mortality"
    out["n"] = len(out)
    out["coverage_count"] = len(out)
    out["missing_count"] = 0
    out["missing_rate"] = 0.0
    out["source_provenance"] = "results/tables/conversion_profiles.csv;results/tables/conversion_bottleneck_scores.csv"
    return out


def _life_variant(root: Path) -> pd.DataFrame:
    path = root / "results/tables/stage4_life_expectancy_conversion_profiles.csv"
    if not path.exists():
        return pd.DataFrame()
    life = pd.read_csv(path)
    if "outcome" not in life.columns or "conversion_profile" not in life.columns:
        return pd.DataFrame()
    primary = life[life["outcome"].eq("max_life_expectancy_drop_2020_2022")].copy()
    if primary.empty:
        primary = life.copy()
    primary["evidence_variant"] = "life_expectancy_loss"
    return primary[
        [
            "iso3",
            "country",
            "evidence_variant",
            "conversion_profile",
            "conversion_efficiency_zscore",
            "realized_resilience_gap",
            "main_bottleneck_block",
            "n",
            "coverage_count",
            "missing_count_from_potential_capacity_sample",
            "missing_rate_from_potential_capacity_sample",
        ]
    ].rename(
        columns={
            "missing_count_from_potential_capacity_sample": "missing_count",
            "missing_rate_from_potential_capacity_sample": "missing_rate",
        }
    ).assign(source_provenance="results/tables/stage4_life_expectancy_conversion_profiles.csv")


def _ghs_variant(root: Path) -> pd.DataFrame:
    path = root / "results/tables/stage4_conversion_profiles_with_ghs.csv"
    if not path.exists():
        return pd.DataFrame()
    ghs = pd.read_csv(path)
    if "conversion_profile" not in ghs.columns:
        return pd.DataFrame()
    ghs = ghs.copy()
    ghs["evidence_variant"] = "ghs_enhanced_owid"
    if "main_bottleneck_block" not in ghs.columns:
        ghs["main_bottleneck_block"] = np.nan
    return ghs[
        [
            "iso3",
            "country",
            "evidence_variant",
            "conversion_profile",
            "conversion_efficiency_zscore",
            "realized_resilience_gap",
            "main_bottleneck_block",
            "n",
            "coverage_count",
            "missing_count_from_potential_capacity_sample",
            "missing_rate_from_potential_capacity_sample",
        ]
    ].rename(
        columns={
            "missing_count_from_potential_capacity_sample": "missing_count",
            "missing_rate_from_potential_capacity_sample": "missing_rate",
        }
    ).assign(source_provenance="results/tables/stage4_conversion_profiles_with_ghs.csv")


def _who_variant(root: Path) -> pd.DataFrame:
    path = root / "results/tables/stage4_who_pulse_profile_overlap.csv"
    if not path.exists():
        return pd.DataFrame()
    who = pd.read_csv(path)
    if "conversion_profile" not in who.columns:
        return pd.DataFrame()
    who = who.copy()
    who["evidence_variant"] = "who_pulse"
    who["main_bottleneck_block"] = np.nan
    who["realized_resilience_gap"] = np.nan
    for col in ["n", "coverage_count", "missing_count", "missing_rate"]:
        if col not in who.columns:
            who[col] = len(who) if col in {"n", "coverage_count"} else 0
    return who[
        [
            "iso3",
            "country",
            "evidence_variant",
            "conversion_profile",
            "conversion_efficiency_zscore",
            "realized_resilience_gap",
            "main_bottleneck_block",
            "n",
            "coverage_count",
            "missing_count",
            "missing_rate",
        ]
    ].assign(source_provenance="results/tables/stage4_who_pulse_profile_overlap.csv")


def _variants(root: Path) -> pd.DataFrame:
    frames = [_baseline_variant(root), _life_variant(root), _ghs_variant(root), _who_variant(root)]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True)


def run_cross_evidence_stability(root: Path = ROOT) -> pd.DataFrame:
    ensure_dirs(root)
    variants = _variants(root)
    save_csv(variants, root / "results/tables/stage4_evidence_variant_country_scores.csv")
    rows = []
    under_rows = []
    over_rows = []
    bottleneck_rows = []
    for a, b in combinations(sorted(variants["evidence_variant"].unique()), 2):
        left = variants[variants["evidence_variant"].eq(a)]
        right = variants[variants["evidence_variant"].eq(b)]
        merged = left.merge(right, on="iso3", suffixes=("_a", "_b"), how="inner")
        if merged.empty:
            continue
        rows.append(
            {
                "evidence_variant_a": a,
                "evidence_variant_b": b,
                "n": len(merged),
                "coverage_count": len(merged),
                "missing_count": int(max(left["coverage_count"].max(), right["coverage_count"].max()) - len(merged)),
                "missing_rate": float(1.0 - len(merged) / max(max(left["coverage_count"].max(), right["coverage_count"].max()), 1)),
                "conversion_efficiency_spearman": spearmanr(
                    merged["conversion_efficiency_zscore_a"], merged["conversion_efficiency_zscore_b"]
                ).statistic,
                "profile_agreement": float((merged["conversion_profile_a"] == merged["conversion_profile_b"]).mean()),
                "stable_label_count": int((merged["conversion_profile_a"] == merged["conversion_profile_b"]).sum()),
                "unstable_label_count": int((merged["conversion_profile_a"] != merged["conversion_profile_b"]).sum()),
                "source_provenance": f"{a};{b}",
            }
        )
        ua = set(left.nsmallest(10, "conversion_efficiency_zscore")["iso3"])
        ub = set(right.nsmallest(10, "conversion_efficiency_zscore")["iso3"])
        oa = set(left.nlargest(10, "conversion_efficiency_zscore")["iso3"])
        ob = set(right.nlargest(10, "conversion_efficiency_zscore")["iso3"])
        under_rows.append(
            {
                "evidence_variant_a": a,
                "evidence_variant_b": b,
                "n": len(merged),
                "coverage_count": len(merged),
                "missing_count": int(max(left["coverage_count"].max(), right["coverage_count"].max()) - len(merged)),
                "missing_rate": float(1.0 - len(merged) / max(max(left["coverage_count"].max(), right["coverage_count"].max()), 1)),
                "top10_under_realizer_overlap": len(ua & ub) / 10,
                "overlap_iso3": ",".join(sorted(ua & ub)),
                "source_provenance": f"{a};{b}",
            }
        )
        over_rows.append(
            {
                "evidence_variant_a": a,
                "evidence_variant_b": b,
                "n": len(merged),
                "coverage_count": len(merged),
                "missing_count": int(max(left["coverage_count"].max(), right["coverage_count"].max()) - len(merged)),
                "missing_rate": float(1.0 - len(merged) / max(max(left["coverage_count"].max(), right["coverage_count"].max()), 1)),
                "top10_over_performer_overlap": len(oa & ob) / 10,
                "overlap_iso3": ",".join(sorted(oa & ob)),
                "source_provenance": f"{a};{b}",
            }
        )
        bottleneck_rows.append(
            {
                "evidence_variant_a": a,
                "evidence_variant_b": b,
                "n": len(merged),
                "coverage_count": len(merged),
                "missing_count": int(max(left["coverage_count"].max(), right["coverage_count"].max()) - len(merged)),
                "missing_rate": float(1.0 - len(merged) / max(max(left["coverage_count"].max(), right["coverage_count"].max()), 1)),
                "bottleneck_block_agreement": float(
                    (merged["main_bottleneck_block_a"] == merged["main_bottleneck_block_b"]).mean()
                ),
                "source_provenance": f"{a};{b}",
            }
        )
    profile_stability = pd.DataFrame(rows)
    under = pd.DataFrame(under_rows)
    over = pd.DataFrame(over_rows)
    bottleneck = pd.DataFrame(bottleneck_rows)
    save_csv(profile_stability, root / "results/tables/stage4_cross_evidence_profile_stability.csv")
    save_csv(under, root / "results/tables/stage4_under_realizer_cross_evidence_overlap.csv")
    save_csv(over, root / "results/tables/stage4_over_performer_cross_evidence_overlap.csv")
    save_csv(bottleneck, root / "results/tables/stage4_bottleneck_cross_evidence_stability.csv")
    country = _country_label_stability(variants, root)
    _plot_profile_agreement(profile_stability, root)
    _plot_country_label_stability(country, root)
    return profile_stability


def _country_label_stability(variants: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    all_variants = sorted(variants["evidence_variant"].unique())
    for iso3, group in variants.groupby("iso3"):
        profiles = group.set_index("evidence_variant")["conversion_profile"].to_dict()
        available = sorted(profiles)
        counts = pd.Series(list(profiles.values())).value_counts()
        top_profile = counts.index[0]
        if len(available) < 2:
            label = "data-limited"
        elif counts.iloc[0] == len(available):
            label = {
                "capacity under-realizers": "stable under-realizer",
                "adaptive over-performers": "stable over-performer",
                "effective converters": "stable effective converter",
                "structurally vulnerable systems": "stable structurally vulnerable",
                "uncertain / data-limited systems": "data-limited",
            }.get(top_profile, "evidence-dependent label")
        else:
            label = "evidence-dependent label"
        rows.append(
            {
                "iso3": iso3,
                "country": group["country"].iloc[0],
                "available_evidence_variants": ",".join(available),
                "n_variants_available": len(available),
                "n_total_variants_considered": len(all_variants),
                "profile_labels_observed": ",".join(sorted(set(profiles.values()))),
                "country_label_category": label,
                "label_stability_rate": float(counts.iloc[0] / len(available)),
                "n": len(available),
                "coverage_count": len(available),
                "missing_count": int(len(all_variants) - len(available)),
                "missing_rate": float(1.0 - len(available) / max(len(all_variants), 1)),
                "source_provenance": "results/tables/stage4_evidence_variant_country_scores.csv",
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/stage4_country_label_stability.csv")
    return out


def _plot_profile_agreement(df: pd.DataFrame, root: Path) -> None:
    if df.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No cross-evidence comparisons available", ha="center", va="center")
        plt.axis("off")
    else:
        labels = df["evidence_variant_a"] + " vs " + df["evidence_variant_b"]
        plt.figure(figsize=(9, max(4, 0.4 * len(df))))
        plt.barh(labels, df["profile_agreement"], color="#2b8cbe")
        plt.xlim(0, 1)
        plt.xlabel("Profile agreement")
        plt.title("Stage 4 cross-evidence profile agreement")
    plt.tight_layout()
    plt.savefig(root / "results/figures/stage4_cross_evidence_profile_agreement.pdf", bbox_inches="tight")
    plt.close()


def _plot_country_label_stability(df: pd.DataFrame, root: Path) -> None:
    counts = df["country_label_category"].value_counts()
    plt.figure(figsize=(8, 4.5))
    plt.barh(counts.index, counts.values, color="#756bb1")
    plt.xlabel("Countries")
    plt.title("Stage 4 country label stability")
    plt.tight_layout()
    plt.savefig(root / "results/figures/stage4_country_label_stability_heatmap.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_cross_evidence_stability()
