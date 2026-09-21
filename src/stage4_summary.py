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


def write_stage4_summary(root: Path = ROOT) -> Path:
    root = Path(root)
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    meta = pd.read_csv(root / "results/tables/expanded_indicator_coverage.csv")
    ghs = pd.read_csv(root / "results/tables/stage4_ghs_2019_integration_audit.csv")
    le_cov = pd.read_csv(root / "results/tables/stage4_life_expectancy_coverage.csv")
    who = pd.read_csv(root / "results/tables/stage4_who_pulse_inventory.csv")
    cross = pd.read_csv(root / "results/tables/stage4_cross_evidence_profile_stability.csv")
    country = pd.read_csv(root / "results/tables/stage4_country_label_stability.csv")
    under = pd.read_csv(root / "results/tables/stage4_case_audit_under_realizers.csv")
    over = pd.read_csv(root / "results/tables/stage4_case_audit_over_performers.csv")
    bottleneck = pd.read_csv(root / "results/tables/stage4_bottleneck_cross_evidence_stability.csv")
    paper_tables = list((root / "paper_assets/tables").glob("*.csv")) if (root / "paper_assets/tables").exists() else []
    paper_figs = list((root / "paper_assets/figures").glob("*.pdf")) if (root / "paper_assets/figures").exists() else []

    stable_count = int(country["country_label_category"].str.startswith("stable").sum())
    evidence_dep = int(country["country_label_category"].eq("evidence-dependent label").sum())
    data_limited = int(country["country_label_category"].eq("data-limited").sum())
    mean_agreement = float(cross["profile_agreement"].mean()) if not cross.empty and "profile_agreement" in cross else 0.0
    life_max_cov = int(le_cov["coverage_count"].max())
    if life_max_cov >= 80 and stable_count >= evidence_dep:
        final = "A. ready for manuscript drafting, with evidence-dependent country labels flagged."
    else:
        final = "B. one more data collection pass needed before drafting."

    lines = [
        "# Run Summary",
        "",
        f"Generated: {now_stamp()}",
        f"Git commit: {_git_hash(root)}",
        "",
        "## Stage 4 Final Evidence Strengthening",
        "",
        "Stage 4 adds only final high-value evidence before manuscript drafting: provenance-controlled GHS integration, programmatic life-expectancy-loss validation, optional WHO pulse validation, cross-evidence stability, stable case audits, and refreshed paper assets.",
        "",
        "## Coverage",
        "",
        f"- Scoring countries: {int(potential['n_countries_in_sample'].max())}",
        f"- Conversion validation countries, OWID baseline: {int(pd.read_csv(root / 'results/tables/conversion_efficiency_scores.csv')['n_validation_countries'].max())}",
        f"- Life expectancy outcome max coverage: {life_max_cov}",
        f"- Usable expanded indicators: {int(meta['use_as_feature_in_expanded_score'].sum())}",
        f"- Feature blocks: {', '.join(sorted(meta.loc[meta['use_as_feature_in_expanded_score'].astype(bool), 'block'].unique()))}",
        "",
        "## Source Status",
        "",
        f"- GHS status: {ghs['status'].iloc[0]}",
        f"- WHO pulse status: {who['status'].iloc[0]}",
        "- Life expectancy loss: downloaded programmatically from World Bank SP.DYN.LE00.IN for 2019-2022.",
        "",
        "## Cross-Evidence Stability",
        "",
        f"- Mean profile agreement across evidence variants: {_fmt(mean_agreement)}",
        f"- Stable country labels: {stable_count}",
        f"- Evidence-dependent labels: {evidence_dep}",
        f"- Data-limited labels: {data_limited}",
        "",
        "Interpretation guardrail: data-limited and evidence-dependent labels are retained as uncertainty outputs. They should be presented as a strength of HSR-Convert, because the system does not force stable country claims when cross-evidence support is weak.",
        "",
        "Stable under-realizers:",
    ]
    if under.empty:
        lines.append("- None selected as stable under-realizers.")
    else:
        for _, row in under.head(5).iterrows():
            lines.append(f"- {row['country']} ({row['iso3']}): E={_fmt(row['conversion_efficiency_zscore'])}, bottleneck={row.get('main_bottleneck_block', 'NA')}")
    lines.append("")
    lines.append("Stable over-performers:")
    if over.empty:
        lines.append("- None selected as stable over-performers.")
    else:
        for _, row in over.head(5).iterrows():
            lines.append(f"- {row['country']} ({row['iso3']}): E={_fmt(row['conversion_efficiency_zscore'])}, bottleneck={row.get('main_bottleneck_block', 'NA')}")
    lines.extend(["", "Stable bottleneck patterns:"])
    if bottleneck.empty:
        lines.append("- No cross-evidence bottleneck comparison available.")
    else:
        lines.append(f"- Mean bottleneck agreement across evidence variants: {_fmt(bottleneck['bottleneck_block_agreement'].mean())}")
    lines.extend(["", "## Paper Assets", "", f"- Final paper tables: {len(paper_tables)}", f"- Final paper figures: {len(paper_figs)}"])
    lines.extend(["", "## Final Evidence Statement", "", f"- {final}", "", "## Skipped Datasets and Why", ""])
    if ghs["status"].iloc[0] == "loaded":
        lines.append(
            f"- GHS 2019 was loaded from `data/raw/ghs/ghs_index_2019.csv`; coverage {int(ghs['coverage_count'].iloc[0])} countries; file hash {ghs['file_hash_sha256'].iloc[0]}."
        )
    else:
        lines.append("- GHS 2019 is skipped unless a legitimate local `data/raw/ghs/ghs_index_2019.csv` file is provided.")
    if who["status"].iloc[0].startswith("skipped"):
        lines.append("- WHO pulse survey is skipped because no legitimate country-level CSV/XLSX/XLS file is present.")
    else:
        lines.append("- WHO pulse survey was loaded from the local country-level file recorded in `stage4_who_pulse_inventory.csv`.")
    lines.extend(
        [
            "",
            "## Reviewer Risk Mitigation",
            "",
            "- Preparedness evidence: GHS 2019 is handled as provenance-controlled preparedness sensitivity evidence; 2021 GHS values are not used as pre-shock features.",
            "- Service-disruption evidence: WHO pulse survey validation remains optional and is skipped unless a legitimate country-level machine-readable file exists. This is a limitation, not a negative service-disruption finding.",
            "- Label uncertainty: stable, evidence-dependent, and data-limited labels are separate outputs. Country-specific Results claims should prioritize stable labels and explicitly flag the other groups.",
            "- Ecological boundary: all analyses are country-level diagnostic associations and must not be written as causal, clinical, emergency-response, or prescriptive policy conclusions.",
            "- Method contribution: the paper should foreground capacity-to-resilience conversion rather than a generic new MCDM framework.",
        ]
    )
    next_step = (
        "Start manuscript drafting. Rerun Stage 4 only if a legitimate WHO pulse country-level file or replacement GHS source is supplied before drafting."
        if ghs["status"].iloc[0] == "loaded"
        else "Start manuscript drafting. Rerun Stage 4 only if legitimate GHS 2019 or WHO pulse survey country-level files are supplied before drafting."
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Life expectancy loss is a secondary health outcome and may reflect broader demographic and reporting factors.",
            "- Cross-evidence labels are diagnostic stability labels, not causal classifications.",
            "- Bottleneck diagnostics are descriptive and should not be written as medical or policy prescriptions.",
            "",
            "## Next Step",
            "",
            next_step,
        ]
    )
    return write_text("\n".join(lines), root / "RUN_SUMMARY.md")


if __name__ == "__main__":
    write_stage4_summary()
