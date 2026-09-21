from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .paper_assets import build_paper_assets
from .utils import ROOT, save_csv, write_text, now_stamp


def refresh_final_paper_assets(root: Path = ROOT) -> None:
    root = Path(root)
    build_paper_assets(root)
    tables = root / "paper_assets/tables"
    figures = root / "paper_assets/figures"
    notes = root / "paper_assets/notes"
    for path in [tables, figures, notes]:
        path.mkdir(parents=True, exist_ok=True)

    cross = pd.read_csv(root / "results/tables/stage4_cross_evidence_profile_stability.csv")
    country = pd.read_csv(root / "results/tables/stage4_country_label_stability.csv")
    table7 = cross.copy()
    table7["stable_country_labels"] = int((country["country_label_category"].str.startswith("stable")).sum())
    table7["evidence_dependent_labels"] = int(country["country_label_category"].eq("evidence-dependent label").sum())
    table7["data_limited_labels"] = int(country["country_label_category"].eq("data-limited").sum())
    table7["asset_source_files"] = "results/tables/stage4_cross_evidence_profile_stability.csv;results/tables/stage4_country_label_stability.csv"
    save_csv(table7, tables / "table7_cross_evidence_stability.csv")

    table8 = pd.read_csv(root / "results/tables/stage4_stable_case_audit.csv")
    table8["asset_source_files"] = "results/tables/stage4_stable_case_audit.csv"
    save_csv(table8, tables / "table8_stable_case_audit.csv")

    shutil.copyfile(root / "results/figures/stage4_cross_evidence_profile_agreement.pdf", figures / "fig7_cross_evidence_stability.pdf")
    shutil.copyfile(root / "results/figures/stage4_stable_case_audit_summary.pdf", figures / "fig8_stable_case_audit.pdf")

    _write_final_notes(root, notes)


def _write_final_notes(root: Path, notes: Path) -> None:
    ghs = pd.read_csv(root / "results/tables/stage4_ghs_2019_integration_audit.csv")
    le = pd.read_csv(root / "results/tables/stage4_life_expectancy_coverage.csv")
    who = pd.read_csv(root / "results/tables/stage4_who_pulse_inventory.csv")
    country = pd.read_csv(root / "results/tables/stage4_country_label_stability.csv")
    stable_labels = int((country["country_label_category"].str.startswith("stable")).sum())
    evidence_dependent = int(country["country_label_category"].eq("evidence-dependent label").sum())
    data_limited = int(country["country_label_category"].eq("data-limited").sum())
    ghs_loaded = ghs["status"].iloc[0] == "loaded"
    who_skipped = who["status"].iloc[0].startswith("skipped")
    rerun_note = (
        "Next step: manuscript drafting. Rerun Stage 4 only if a legitimate WHO pulse file or replacement GHS source is supplied.\n"
        if ghs_loaded
        else "Next step: manuscript drafting, unless the user supplies legitimate GHS or WHO pulse files for one final rerun.\n"
    )
    write_text(
        "# Final Experiment Status\n\n"
        f"Generated: {now_stamp()}\n\n"
        f"- GHS 2019 status: {ghs['status'].iloc[0]}\n"
        f"- Life expectancy outcomes: {len(le)} variants, max coverage {int(le['coverage_count'].max())} countries\n"
        f"- WHO pulse status: {who['status'].iloc[0]}\n"
        f"- Stable labels: {stable_labels}\n"
        f"- Evidence-dependent labels: {evidence_dependent}\n"
        f"- Data-limited labels: {data_limited}\n\n"
        f"{rerun_note}",
        notes / "final_experiment_status.md",
    )
    write_text(
        "# Claims We Can Make\n\n"
        "- HSR-Convert separates pre-shock potential capacity from shock-realized conversion efficiency.\n"
        "- The methodological contribution is the capacity-to-resilience conversion layer, not a generic new MCDM ranking method.\n"
        "- Official GHS 2019 evidence is integrated as a provenance-controlled preparedness sensitivity when loaded; 2021 GHS values are not used as pre-shock features.\n"
        "- Life expectancy loss provides a programmatic secondary outcome for cross-evidence stability checks.\n"
        "- Stable country labels can be distinguished from evidence-dependent and data-limited labels.\n"
        "- Data-limited labels are an explicit uncertainty output: HSR-Convert does not force country claims when evidence support is weak.\n"
        "- WHO pulse evidence is provenance-controlled; if no legitimate country-level file is present, the paper can report that service-disruption validation was not performed.\n",
        notes / "claims_we_can_make.md",
    )
    write_text(
        "# Claims We Must Not Make\n\n"
        "- Do not make causal claims.\n"
        "- Do not write country-level diagnostics as clinical advice, emergency triage, or causal policy prescriptions.\n"
        "- Do not claim WHO pulse service-disruption evidence was used when no legitimate country-level file was available.\n"
        "- Do not use GHS 2021 values as pre-shock features.\n"
        "- Do not treat country labels that vary across evidence variants as stable.\n"
        "- Do not hide data-limited countries or convert them into firm under-realizer or over-performer claims.\n"
        "- Do not frame the contribution as acceptance-worthy because it uses many MCDM or machine-learning methods.\n"
        "- Do not make medical or policy prescriptions from bottleneck diagnostics.\n",
        notes / "claims_we_must_not_make.md",
    )
    write_text(
        "# Reviewer Risk Mitigation\n\n"
        "## Preparedness and Service-Disruption Evidence\n\n"
        + (
            f"- GHS 2019 is loaded from the audited official raw CSV with {int(ghs['coverage_count'].iloc[0])} matched scoring countries.\n"
            if ghs_loaded
            else "- GHS 2019 is not used because no legitimate local file was available; skipped status is explicit.\n"
        )
        + (
            "- WHO pulse survey remains skipped because no legitimate country-level machine-readable file is present. This is a limitation of service-disruption validation, not evidence of no disruption.\n"
            if who_skipped
            else "- WHO pulse survey is loaded from the local country-level file recorded in the inventory.\n"
        )
        + "\n## Label Uncertainty\n\n"
        f"- Stable labels: {stable_labels}; evidence-dependent labels: {evidence_dependent}; data-limited labels: {data_limited}.\n"
        "- The large data-limited group should be framed as a diagnostic safeguard: the system refuses to over-classify countries with insufficient cross-evidence support.\n\n"
        "## Ecological Analysis Boundary\n\n"
        "- All outputs are country-level diagnostic associations. They are not individual-level, clinical, emergency-response, or causal policy estimates.\n"
        "- Life expectancy loss is a secondary outcome and may reflect broader health-system, demographic, and reporting factors beyond COVID-period mortality.\n\n"
        "## Method Contribution\n\n"
        "- The core contribution is capacity-to-resilience conversion for health-system resilience decision support.\n"
        "- The paper should emphasize a problem-specific decision-support architecture: pre-shock capacity scoring, expected shock-burden modeling, realized resilience gaps, conversion efficiency, profile stability, and bottleneck diagnostics.\n"
        "- MCDM and ML components are supporting modules inside HSR-Convert, not the claimed contribution by themselves.\n",
        notes / "reviewer_risk_mitigation.md",
    )


if __name__ == "__main__":
    refresh_final_paper_assets()
