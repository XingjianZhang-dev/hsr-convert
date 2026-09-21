from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_required_stage4_outputs_exist_after_stage4_run() -> None:
    marker = ROOT / "results/tables/stage4_life_expectancy_coverage.csv"
    if not marker.exists():
        return
    required = [
        "results/tables/stage4_ghs_2019_integration_audit.csv",
        "results/tables/stage4_preparedness_block_with_ghs.csv",
        "results/tables/stage4_ghs_benchmark_comparison.csv",
        "results/tables/stage4_conversion_profiles_with_ghs.csv",
        "results/tables/stage4_profile_shift_after_ghs.csv",
        "results/tables/stage4_life_expectancy_coverage.csv",
        "results/tables/stage4_life_expectancy_secondary_validation.csv",
        "results/tables/stage4_life_expectancy_conversion_profiles.csv",
        "results/tables/stage4_life_expectancy_under_over_performers.csv",
        "results/tables/stage4_who_pulse_inventory.csv",
        "results/tables/stage4_who_pulse_secondary_validation.csv",
        "results/tables/stage4_who_pulse_profile_overlap.csv",
        "results/tables/stage4_cross_evidence_profile_stability.csv",
        "results/tables/stage4_under_realizer_cross_evidence_overlap.csv",
        "results/tables/stage4_over_performer_cross_evidence_overlap.csv",
        "results/tables/stage4_bottleneck_cross_evidence_stability.csv",
        "results/tables/stage4_country_label_stability.csv",
        "results/tables/stage4_stable_case_audit.csv",
        "results/tables/stage4_case_audit_under_realizers.csv",
        "results/tables/stage4_case_audit_over_performers.csv",
        "results/tables/stage4_case_audit_uncertain_labels.csv",
        "results/figures/stage4_preparedness_before_after_ghs.pdf",
        "results/figures/stage4_ghs_vs_conversion_efficiency.pdf",
        "results/figures/stage4_life_expectancy_loss_distribution.pdf",
        "results/figures/stage4_conversion_vs_life_expectancy_loss.pdf",
        "results/figures/stage4_who_pulse_profile_validation.pdf",
        "results/figures/stage4_cross_evidence_profile_agreement.pdf",
        "results/figures/stage4_country_label_stability_heatmap.pdf",
        "results/figures/stage4_stable_case_audit_summary.pdf",
        "RUN_SUMMARY.md",
        "METHODS.md",
        "DATA_CARD.md",
        "DATA_SOURCES.md",
        "RESULTS_AUDIT.md",
        "CONCEPTUAL_FRAMEWORK.md",
        "README.md",
    ]
    missing = [rel for rel in required if not (ROOT / rel).exists()]
    assert not missing
    empty = [rel for rel in required if (ROOT / rel).is_file() and (ROOT / rel).stat().st_size == 0]
    assert not empty


def test_stage4_tables_include_coverage_and_missingness_after_run() -> None:
    marker = ROOT / "results/tables/stage4_life_expectancy_coverage.csv"
    if not marker.exists():
        return
    for path in sorted((ROOT / "results/tables").glob("stage4_*.csv")):
        df = pd.read_csv(path)
        assert "coverage_count" in df.columns, path.name
        assert any(col in df.columns for col in ["n", "n_scoring_countries"]), path.name
        assert any(col in df.columns for col in ["missing_count", "missing_count_from_potential_capacity_sample"]), path.name
        assert any(col in df.columns for col in ["missing_rate", "missing_rate_from_potential_capacity_sample"]), path.name
        assert any(col in df.columns for col in ["source_provenance", "source_indicator", "file_path", "stage"]), path.name


def test_configured_pre_shock_features_remain_pre_2020() -> None:
    for config_name in ["indicators.yml", "indicator_blocks.yml"]:
        cfg = yaml.safe_load((ROOT / f"config/{config_name}").read_text(encoding="utf-8"))
        for indicator in cfg["indicators"]:
            if indicator.get("use_as_feature", False):
                assert max(indicator["years_used"]) < 2020, indicator["variable"]
