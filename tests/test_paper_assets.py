from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_paper_assets_exist_and_reference_result_sources_after_run() -> None:
    table = ROOT / "paper_assets/tables/table1_data_sources_and_feature_blocks.csv"
    if not table.exists():
        return
    required_tables = [
        "table1_data_sources_and_feature_blocks.csv",
        "table2_conversion_model_diagnostics.csv",
        "table3_conversion_profiles_and_examples.csv",
        "table4_bottleneck_diagnostics.csv",
        "table5_block_ablation_and_benchmarks.csv",
        "table6_robustness_summary.csv",
    ]
    for name in required_tables:
        path = ROOT / "paper_assets/tables" / name
        assert path.exists()
        df = pd.read_csv(path)
        assert "asset_source_files" in df.columns
    required_figures = [
        "fig1_hsr_convert_framework.pdf",
        "fig2_potential_capacity_vs_shock_burden.pdf",
        "fig3_conversion_efficiency_profiles.pdf",
        "fig4_bottleneck_heatmap.pdf",
        "fig5_block_ablation_benchmark.pdf",
        "fig6_conversion_uncertainty_intervals.pdf",
    ]
    for name in required_figures:
        assert (ROOT / "paper_assets/figures" / name).exists()
