from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .utils import ROOT, ensure_dirs, now_stamp, save_csv, write_text


ASSET_ROOT = "paper_assets"


def _asset_dirs(root: Path) -> tuple[Path, Path, Path]:
    base = root / ASSET_ROOT
    tables = base / "tables"
    figures = base / "figures"
    notes = base / "notes"
    for path in [tables, figures, notes]:
        path.mkdir(parents=True, exist_ok=True)
    return tables, figures, notes


def build_paper_assets(root: Path = ROOT) -> None:
    ensure_dirs(root)
    tables_dir, figures_dir, notes_dir = _asset_dirs(root)
    _table1(root, tables_dir)
    _table2(root, tables_dir)
    _table3(root, tables_dir)
    _table4(root, tables_dir)
    _table5(root, tables_dir)
    _table6(root, tables_dir)
    _figures(root, figures_dir)
    _notes(root, notes_dir)


def _table1(root: Path, out: Path) -> None:
    meta = pd.read_csv(root / "results/tables/expanded_indicator_coverage.csv")
    block_missing = pd.read_csv(root / "results/tables/block_missingness.csv")
    table = (
        meta[meta["use_as_feature_in_expanded_score"].astype(bool)]
        .groupby(["block", "source"], as_index=False)
        .agg(
            n_indicators=("variable", "count"),
            n_countries_median=("n_countries", "median"),
            missing_rate_median=("missing_rate", "median"),
            variables=("variable", lambda x: ",".join(x)),
        )
    )
    table = table.merge(block_missing[["block", "n_countries", "median_country_missing_rate"]], on="block", how="left")
    table["asset_source_files"] = "results/tables/expanded_indicator_coverage.csv;results/tables/block_missingness.csv"
    save_csv(table, out / "table1_data_sources_and_feature_blocks.csv")


def _table2(root: Path, out: Path) -> None:
    diag = pd.read_csv(root / "results/tables/expected_shock_burden_model_diagnostics.csv")
    agreement = pd.read_csv(root / "results/tables/conversion_model_agreement.csv")
    table = diag.copy()
    table["model_agreement_mean_spearman"] = float(agreement["spearman_efficiency"].mean())
    table["model_agreement_min_spearman"] = float(agreement["spearman_efficiency"].min())
    table["model_agreement_mean_profile_agreement"] = float(agreement["profile_assignment_agreement"].mean())
    table["asset_source_files"] = "results/tables/expected_shock_burden_model_diagnostics.csv;results/tables/conversion_model_agreement.csv"
    save_csv(table, out / "table2_conversion_model_diagnostics.csv")


def _table3(root: Path, out: Path) -> None:
    profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    cases = pd.read_csv(root / "results/tables/conversion_case_studies.csv")
    counts = profiles.groupby("conversion_profile", as_index=False).agg(
        n=("iso3", "nunique"),
        median_efficiency=("conversion_efficiency_zscore", "median"),
        median_gap=("realized_resilience_gap", "median"),
        missing_rate_mean=("expanded_feature_missing_rate", "mean"),
    )
    examples = cases.groupby("conversion_profile")["country"].apply(lambda x: ", ".join(x.head(3))).reset_index(name="example_countries")
    table = counts.merge(examples, on="conversion_profile", how="left")
    table["coverage_count"] = profiles["iso3"].nunique()
    table["missing_count"] = 0
    table["missing_rate"] = 0.0
    table["asset_source_files"] = "results/tables/conversion_profiles.csv;results/tables/conversion_case_studies.csv"
    save_csv(table, out / "table3_conversion_profiles_and_examples.csv")


def _table4(root: Path, out: Path) -> None:
    bottleneck = pd.read_csv(root / "results/tables/conversion_bottleneck_scores.csv")
    table = bottleneck.groupby(["conversion_profile", "weakest_bottleneck_block"], as_index=False).agg(
        n=("iso3", "nunique"),
        median_bottleneck_gap=("weakest_bottleneck_gap", "median"),
        median_efficiency=("conversion_efficiency_zscore", "median"),
        coverage_count=("n_validation_countries", "max"),
    )
    table["missing_count"] = 0
    table["missing_rate"] = 0.0
    table["asset_source_files"] = "results/tables/conversion_bottleneck_scores.csv"
    save_csv(table, out / "table4_bottleneck_diagnostics.csv")


def _table5(root: Path, out: Path) -> None:
    ablation = pd.read_csv(root / "results/tables/block_ablation_validation.csv")
    bench = pd.read_csv(root / "results/tables/benchmark_comparison.csv")
    table = pd.concat(
        [
            ablation.assign(section="block_ablation"),
            bench.rename(columns={"adds_hsr_convert_capability": "interpretation_note"}).assign(section="benchmark"),
        ],
        ignore_index=True,
        sort=False,
    )
    table["asset_source_files"] = "results/tables/block_ablation_validation.csv;results/tables/benchmark_comparison.csv"
    save_csv(table, out / "table5_block_ablation_and_benchmarks.csv")


def _table6(root: Path, out: Path) -> None:
    uncertainty = pd.read_csv(root / "results/tables/conversion_efficiency_uncertainty.csv")
    mc = pd.read_csv(root / "results/tables/conversion_monte_carlo_summary.csv")
    rf = pd.read_csv(root / "results/tables/rf_benchmark_audit.csv")
    ghs = pd.read_csv(root / "results/tables/ghs_2019_integration_audit.csv")
    secondary = pd.read_csv(root / "results/tables/secondary_outcome_inventory.csv")
    table = pd.DataFrame(
        [
            {
                "robustness_domain": "bootstrap_conversion_uncertainty",
                "n": int(uncertainty["coverage_count"].max()),
                "coverage_count": int(uncertainty["coverage_count"].max()),
                "missing_count": int(uncertainty["missing_count_from_potential_capacity_sample"].max()),
                "missing_rate": float(uncertainty["missing_rate_from_potential_capacity_sample"].max()),
                "summary_metric": "median_profile_stability",
                "summary_value": float(uncertainty["profile_stability_score"].median()),
            },
            {
                "robustness_domain": "monte_carlo_conversion_uncertainty",
                "n": int(mc["coverage_count"].max()),
                "coverage_count": int(mc["coverage_count"].max()),
                "missing_count": int(mc["missing_count_from_potential_capacity_sample"].max()),
                "missing_rate": float(mc["missing_rate_from_potential_capacity_sample"].max()),
                "summary_metric": "robust_profile_share",
                "summary_value": float((mc["robustness_category"] == "robust").mean()),
            },
            {
                "robustness_domain": "random_forest_sensitivity",
                "n": int(rf["coverage_count"].max()),
                "coverage_count": int(rf["coverage_count"].max()),
                "missing_count": int(rf["missing_count"].max()),
                "missing_rate": float(rf["missing_rate"].max()),
                "summary_metric": "no_leakage_detected",
                "summary_value": bool(rf["no_leakage_detected"].all()),
            },
            {
                "robustness_domain": "ghs_2019_integration",
                "n": int(ghs["coverage_count"].max()),
                "coverage_count": int(ghs["coverage_count"].max()),
                "missing_count": int(ghs["missing_count"].max()),
                "missing_rate": float(ghs["missing_rate"].max()),
                "summary_metric": "status",
                "summary_value": ghs["status"].iloc[0],
            },
            {
                "robustness_domain": "secondary_outcome_validation",
                "n": int(secondary["coverage_count"].max()),
                "coverage_count": int(secondary["coverage_count"].max()),
                "missing_count": int(secondary["missing_count_from_scoring_sample"].max()),
                "missing_rate": float(secondary["missing_rate_from_scoring_sample"].max()),
                "summary_metric": "status",
                "summary_value": secondary["status"].iloc[0],
            },
        ]
    )
    table["asset_source_files"] = "Stage 3 robustness result tables"
    save_csv(table, out / "table6_robustness_summary.csv")


def _figures(root: Path, out: Path) -> None:
    _framework_figure(out / "fig1_hsr_convert_framework.pdf")
    copies = {
        "results/figures/potential_capacity_vs_shock_burden.pdf": "fig2_potential_capacity_vs_shock_burden.pdf",
        "results/figures/conversion_efficiency_quadrants.pdf": "fig3_conversion_efficiency_profiles.pdf",
        "results/figures/conversion_bottleneck_heatmap.pdf": "fig4_bottleneck_heatmap.pdf",
        "results/figures/benchmark_comparison.pdf": "fig5_block_ablation_benchmark.pdf",
        "results/figures/conversion_monte_carlo_intervals.pdf": "fig6_conversion_uncertainty_intervals.pdf",
    }
    for src, dst in copies.items():
        shutil.copyfile(root / src, out / dst)


def _framework_figure(path: Path) -> None:
    plt.figure(figsize=(10, 4))
    nodes = [
        ("Potential\nCapacity", 0.08),
        ("Expected\nShock Burden", 0.28),
        ("Observed\nShock Burden", 0.48),
        ("Realized\nGap", 0.66),
        ("Conversion\nEfficiency", 0.84),
    ]
    for label, x in nodes:
        plt.text(x, 0.55, label, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.35", fc="#edf8fb", ec="#2b8cbe"))
    for (_, x1), (_, x2) in zip(nodes[:-1], nodes[1:]):
        plt.annotate("", xy=(x2 - 0.08, 0.55), xytext=(x1 + 0.08, 0.55), arrowprops=dict(arrowstyle="->", lw=1.5))
    plt.text(0.5, 0.17, "Pre-shock indicators stay separate from shock-period validation outcomes until conversion modeling.", ha="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _notes(root: Path, out: Path) -> None:
    summary = (root / "RUN_SUMMARY.md").read_text(encoding="utf-8")
    write_text(
        "# Main Findings\n\n"
        "HSR-Convert remains framed as a capacity-to-resilience conversion framework. The evidence package reports potential capacity, expected burden, realized gap, conversion efficiency, profiles, bottlenecks, and robustness diagnostics.\n",
        out / "main_findings.md",
    )
    write_text(
        "# Limitations\n\n"
        "- Excess mortality coverage is incomplete and may reflect mortality-registration capacity.\n"
        "- Expected-burden models are predictive association models, not causal models.\n"
        "- GHS and secondary outcomes are used only when legitimate local files are available.\n",
        out / "limitations.md",
    )
    write_text(
        "# Positioning\n\n"
        "The contribution is an uncertainty-aware decision-support system that diagnoses conversion from pre-shock capacity to shock-realized resilience, rather than a generic MCDM ranking paper.\n",
        out / "positioning.md",
    )
    write_text(
        "# Claims We Can Make\n\n"
        "- HSR-Convert quantifies potential capacity, expected burden, realized gap, and conversion efficiency.\n"
        "- Robustness outputs identify stable and unstable country diagnostics.\n"
        "- Bottleneck outputs are diagnostic decision-support signals.\n\n"
        f"Summary source timestamp: {now_stamp()}\n",
        out / "claims_we_can_make.md",
    )
    write_text(
        "# Claims We Must Not Make\n\n"
        "- Do not claim causal effects.\n"
        "- Do not claim static capacity scores alone fully predict shock outcomes.\n"
        "- Do not claim GHS or secondary outcomes were used if their legitimate files were unavailable.\n"
        "- Do not present RF sensitivity as the main model.\n",
        out / "claims_we_must_not_make.md",
    )
    write_text(summary, out / "run_summary_snapshot.md")


if __name__ == "__main__":
    build_paper_assets()
