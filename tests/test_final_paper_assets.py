from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_final_paper_assets_include_stage4_tables_and_figures() -> None:
    marker = ROOT / "paper_assets/tables/table7_cross_evidence_stability.csv"
    if not marker.exists():
        return
    required_tables = [
        "table1_data_sources_and_feature_blocks.csv",
        "table2_conversion_model_diagnostics.csv",
        "table3_conversion_profiles_and_examples.csv",
        "table4_bottleneck_diagnostics.csv",
        "table5_block_ablation_and_benchmarks.csv",
        "table6_robustness_summary.csv",
        "table7_cross_evidence_stability.csv",
        "table8_stable_case_audit.csv",
    ]
    for name in required_tables:
        path = ROOT / "paper_assets/tables" / name
        assert path.exists(), name
        table = pd.read_csv(path)
        assert "asset_source_files" in table.columns, name

    required_figures = [
        "fig1_hsr_convert_framework.pdf",
        "fig2_potential_capacity_vs_shock_burden.pdf",
        "fig3_conversion_efficiency_profiles.pdf",
        "fig4_bottleneck_heatmap.pdf",
        "fig5_block_ablation_benchmark.pdf",
        "fig6_conversion_uncertainty_intervals.pdf",
        "fig7_cross_evidence_stability.pdf",
        "fig8_stable_case_audit.pdf",
    ]
    for name in required_figures:
        path = ROOT / "paper_assets/figures" / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def test_final_notes_record_stage4_claim_boundaries() -> None:
    marker = ROOT / "paper_assets/notes/final_experiment_status.md"
    if not marker.exists():
        return
    required_notes = [
        "main_findings.md",
        "positioning.md",
        "claims_we_can_make.md",
        "claims_we_must_not_make.md",
        "limitations.md",
        "final_experiment_status.md",
        "reviewer_risk_mitigation.md",
    ]
    for name in required_notes:
        assert (ROOT / "paper_assets/notes" / name).exists(), name
    claims_not = (ROOT / "paper_assets/notes/claims_we_must_not_make.md").read_text(encoding="utf-8").lower()
    assert "causal" in claims_not


def test_final_notes_include_reviewer_risk_guardrails() -> None:
    path = ROOT / "paper_assets/notes/reviewer_risk_mitigation.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").lower()
    for phrase in [
        "who pulse",
        "data-limited",
        "ecological",
        "not individual-level",
        "capacity-to-resilience conversion",
        "mcdm and ml components are supporting modules",
    ]:
        assert phrase in text
