from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ROOT, ensure_dirs, save_csv


def _diagnostic_note(row: pd.Series) -> str:
    blocks = {
        "capacity": row.get("capacity_block_score"),
        "preparedness": row.get("preparedness_block_score"),
        "vulnerability": row.get("vulnerability_block_score"),
        "equity/access": row.get("equity_access_block_score"),
    }
    weakest = min(blocks, key=lambda k: blocks[k] if pd.notna(blocks[k]) else 999)
    return (
        f"Diagnostic profile: {row['profile_label']}; weakest observed block is {weakest}; "
        f"main bottleneck signal is {row['main_bottleneck_block']}. Descriptive only, no causal claim."
    )


def run_stable_case_audit(root: Path = ROOT) -> pd.DataFrame:
    ensure_dirs(root)
    profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    eff = pd.read_csv(root / "results/tables/conversion_efficiency_scores.csv")
    uncertainty = pd.read_csv(root / "results/tables/conversion_efficiency_uncertainty.csv")
    label = pd.read_csv(root / "results/tables/stage4_country_label_stability.csv")
    bottleneck = pd.read_csv(root / "results/tables/conversion_bottleneck_scores.csv")
    bottleneck_prob = pd.read_csv(root / "results/tables/bottleneck_stability_table.csv")
    life_profiles = pd.read_csv(root / "results/tables/stage4_life_expectancy_conversion_profiles.csv")
    life_available = set(life_profiles["iso3"]) if "iso3" in life_profiles.columns else set()

    main_bottleneck_prob = (
        bottleneck_prob.sort_values("bottleneck_probability", ascending=False)
        .drop_duplicates("iso3")[["iso3", "bottleneck_block", "bottleneck_probability"]]
        .rename(columns={"bottleneck_block": "main_bottleneck_block_mc", "bottleneck_probability": "bottleneck_probability"})
    )
    base = profiles.merge(
        eff[
            [
                "iso3",
                "observed_shock_burden",
                "predicted_shock_burden",
                "realized_resilience_gap",
                "potential_capacity_score",
            ]
        ],
        on="iso3",
        how="left",
        suffixes=("", "_eff"),
    )
    base = base.merge(
        uncertainty[
            [
                "iso3",
                "profile_stability_score",
                "conversion_efficiency_ci_low",
                "conversion_efficiency_ci_high",
            ]
        ],
        on="iso3",
        how="left",
    )
    base = base.merge(
        bottleneck[["iso3", "weakest_bottleneck_block"]].rename(columns={"weakest_bottleneck_block": "main_bottleneck_block"}),
        on="iso3",
        how="left",
    )
    base = base.merge(main_bottleneck_prob, on="iso3", how="left")
    base = base.merge(label[["iso3", "country_label_category", "label_stability_rate"]], on="iso3", how="left")
    base["profile_label"] = base["conversion_profile"]
    base["profile_stability_probability"] = base["profile_stability_score"]
    base["expected_shock_burden"] = base["predicted_shock_burden"]
    base["validation_outcomes_available"] = base["iso3"].apply(
        lambda iso: "OWID excess mortality; life expectancy loss" if iso in life_available else "OWID excess mortality"
    )
    base = base.rename(
        columns={
            "capacity_score": "capacity_block_score",
            "preparedness_score": "preparedness_block_score",
            "vulnerability_score": "vulnerability_block_score",
            "equity_access_score": "equity_access_block_score",
            "expanded_feature_missing_rate": "missingness_rate",
        }
    )

    selections = []
    rules = [
        ("stable under-realizer", "conversion_efficiency_zscore", True, 5),
        ("stable over-performer", "conversion_efficiency_zscore", False, 5),
        ("stable effective converter", "conversion_efficiency_zscore", False, 4),
        ("stable structurally vulnerable", "conversion_efficiency_zscore", True, 4),
        ("evidence-dependent label", "label_stability_rate", True, 4),
        ("data-limited", "label_stability_rate", True, 4),
    ]
    for label_name, sort_col, asc, n in rules:
        sub = base[base["country_label_category"].eq(label_name)].copy()
        if sub.empty:
            if label_name == "stable under-realizer":
                sub = base.nsmallest(n, "conversion_efficiency_zscore")
            elif label_name == "stable over-performer":
                sub = base.nlargest(n, "conversion_efficiency_zscore")
            elif label_name == "data-limited":
                sub = base.sort_values("method_disagreement_index", ascending=False).head(n)
            else:
                continue
        else:
            sub = sub.sort_values(sort_col, ascending=asc).head(n)
        selections.append(sub)
    audit = pd.concat(selections, ignore_index=True).drop_duplicates("iso3")
    audit["diagnostic_note"] = audit.apply(_diagnostic_note, axis=1)
    audit["n"] = len(audit)
    audit["coverage_count"] = len(audit)
    audit["missing_count"] = 0
    audit["missing_rate"] = 0.0
    audit["source_provenance"] = (
        "results/tables/conversion_profiles.csv;results/tables/conversion_efficiency_scores.csv;"
        "results/tables/conversion_efficiency_uncertainty.csv;results/tables/stage4_country_label_stability.csv"
    )
    cols = [
        "iso3",
        "country",
        "profile_label",
        "profile_stability_score",
        "profile_stability_probability",
        "conversion_efficiency_zscore",
        "conversion_efficiency_ci_low",
        "conversion_efficiency_ci_high",
        "observed_shock_burden",
        "predicted_shock_burden",
        "expected_shock_burden",
        "realized_resilience_gap",
        "potential_capacity_score",
        "capacity_block_score",
        "preparedness_block_score",
        "vulnerability_block_score",
        "equity_access_block_score",
        "main_bottleneck_block",
        "bottleneck_probability",
        "missingness_rate",
        "validation_outcomes_available",
        "country_label_category",
        "label_stability_rate",
        "diagnostic_note",
        "n",
        "coverage_count",
        "missing_count",
        "missing_rate",
        "source_provenance",
    ]
    audit = audit[[c for c in cols if c in audit.columns]].copy()
    save_csv(audit, root / "results/tables/stage4_stable_case_audit.csv")
    save_csv(
        audit[audit["country_label_category"].eq("stable under-realizer")],
        root / "results/tables/stage4_case_audit_under_realizers.csv",
    )
    save_csv(
        audit[audit["country_label_category"].eq("stable over-performer")],
        root / "results/tables/stage4_case_audit_over_performers.csv",
    )
    save_csv(
        audit[audit["country_label_category"].isin(["evidence-dependent label", "data-limited"])],
        root / "results/tables/stage4_case_audit_uncertain_labels.csv",
    )
    _plot_case_audit(audit, root)
    return audit


def _plot_case_audit(audit: pd.DataFrame, root: Path) -> None:
    counts = audit["country_label_category"].value_counts()
    plt.figure(figsize=(8, 4.5))
    plt.barh(counts.index, counts.values, color="#3182bd")
    plt.xlabel("Selected case countries")
    plt.title("Stage 4 stable case audit summary")
    plt.tight_layout()
    plt.savefig(root / "results/figures/stage4_stable_case_audit_summary.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_stable_case_audit()
