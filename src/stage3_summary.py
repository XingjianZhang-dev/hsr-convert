from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from .utils import ROOT, now_stamp, write_text


def _git_hash(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "not available"


def _fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_stage3_summary(root: Path = ROOT) -> Path:
    root = Path(root)
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    meta = pd.read_csv(root / "results/tables/expanded_indicator_coverage.csv")
    model_agree = pd.read_csv(root / "results/tables/conversion_model_agreement.csv")
    boot = pd.read_csv(root / "results/tables/conversion_efficiency_uncertainty.csv")
    mc = pd.read_csv(root / "results/tables/conversion_monte_carlo_summary.csv")
    profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    bottleneck = pd.read_csv(root / "results/tables/bottleneck_stability_table.csv")
    rf = pd.read_csv(root / "results/tables/rf_benchmark_audit.csv")
    ghs = pd.read_csv(root / "results/tables/ghs_2019_integration_audit.csv")
    secondary = pd.read_csv(root / "results/tables/secondary_outcome_inventory.csv")
    paper_tables = sorted((root / "paper_assets/tables").glob("*.csv")) if (root / "paper_assets/tables").exists() else []
    paper_figures = sorted((root / "paper_assets/figures").glob("*.pdf")) if (root / "paper_assets/figures").exists() else []

    stable_under = boot.sort_values("probability_under_realizer", ascending=False).head(5)
    stable_over = boot.sort_values("probability_over_performer", ascending=False).head(5)
    bottleneck_summary = bottleneck.groupby("bottleneck_block")["bottleneck_probability"].mean().sort_values(ascending=False)
    rf_main = rf[rf["audit_item"].eq("random_forest_repeated_kfold_cv")].head(1)
    rf_line = "RF repeated-CV audit unavailable."
    if not rf_main.empty:
        row = rf_main.iloc[0]
        rf_line = (
            f"RF sensitivity repeated K-fold RMSE={_fmt(row['rmse_mean'], 1)} "
            f"(std {_fmt(row['rmse_std'], 1)}), no leakage detected={bool(row['no_leakage_detected'])}."
        )
    model_line = (
        f"Model agreement mean Spearman={_fmt(model_agree['spearman_efficiency'].mean())}, "
        f"minimum Spearman={_fmt(model_agree['spearman_efficiency'].min())}, "
        f"mean profile agreement={_fmt(model_agree['profile_assignment_agreement'].mean())}."
    )
    robust_share = float((mc["robustness_category"] == "robust").mean())
    median_boot_stability = float(boot["profile_stability_score"].median())
    if robust_share >= 0.60 and median_boot_stability >= 0.60:
        evidence = "A. HSR-Convert is robust enough to start manuscript drafting, with unstable country labels flagged."
    else:
        evidence = "B. HSR-Convert needs more data/outcomes before manuscript drafting."

    lines = [
        "# Run Summary",
        "",
        f"Generated: {now_stamp()}",
        f"Git commit: {_git_hash(root)}",
        "",
        "## Stage 3 Reviewer-Proof Evidence Package",
        "",
        "HSR-Convert measures how pre-shock health-system capacity is converted into shock-realized resilience. Stage 3 adds robustness, uncertainty, provenance, RF sensitivity auditing, and paper-ready evidence assets.",
        "",
        "## Coverage",
        "",
        f"- Expanded scoring countries: {int(potential['n_countries_in_sample'].max())}",
        f"- Validation countries in conversion models: {int(boot['coverage_count'].max())}",
        f"- Usable expanded indicators: {int(meta['use_as_feature_in_expanded_score'].sum())}",
        f"- Feature blocks: {', '.join(sorted(meta.loc[meta['use_as_feature_in_expanded_score'].astype(bool), 'block'].unique()))}",
        "",
        "## GHS and Secondary Outcomes",
        "",
        f"- GHS 2019 status: {ghs['status'].iloc[0]}",
        f"- Secondary outcome status: {secondary['status'].iloc[0]}",
        "",
        "## Robustness Findings",
        "",
        f"- {model_line}",
        f"- Bootstrap median profile stability score: {_fmt(median_boot_stability)}",
        f"- Monte Carlo robust-profile share: {_fmt(robust_share)}",
        "",
        "Stable under-realizer candidates:",
    ]
    for _, row in stable_under.iterrows():
        lines.append(f"- {row['country']} ({row['iso3']}): P(under-realizer)={_fmt(row['probability_under_realizer'])}")
    lines.append("")
    lines.append("Stable over-performer candidates:")
    for _, row in stable_over.iterrows():
        lines.append(f"- {row['country']} ({row['iso3']}): P(over-performer)={_fmt(row['probability_over_performer'])}")
    lines.extend(["", "Stable bottleneck patterns:"])
    for block, prob in bottleneck_summary.items():
        lines.append(f"- {block}: mean bottleneck probability {_fmt(prob)}")
    lines.extend(
        [
            "",
            "## Random Forest Audit",
            "",
            f"- {rf_line}",
            "- RF remains a sensitivity benchmark, not the main HSR-Convert model.",
            "",
            "## Paper Assets",
            "",
            f"- Paper tables generated: {len(paper_tables)}",
            f"- Paper figures generated: {len(paper_figures)}",
            "",
            "## Final Evidence Statement",
            "",
            f"- {evidence}",
            "",
            "## Skipped Datasets and Failed Experiments",
            "",
            "- GHS 2019 integration is skipped unless a legitimate local file is provided.",
            "- Secondary outcomes are skipped unless legitimate country-level files are provided.",
            "- Optional profile clustering was not run because Stage 3 focused on hardening the rule-based HSR-Convert framework.",
            "",
            "## Limitations",
            "",
            "- Excess mortality coverage remains incomplete.",
            "- Conversion models are predictive association models and do not support causal policy claims.",
            "- Country labels with low stability should be described as uncertain.",
            "- RF sensitivity must not be treated as the primary evidence.",
            "",
            "## Next Recommended Experiment",
            "",
            "Add legitimate GHS 2019 and WHO pulse survey country-level data, then rerun Stage 3 to test whether preparedness and service-disruption evidence changes conversion profiles.",
        ]
    )
    return write_text("\n".join(lines), root / "RUN_SUMMARY.md")
