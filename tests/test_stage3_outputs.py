from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_stage3_outputs_exist_after_stage3_run() -> None:
    marker = ROOT / "results/tables/conversion_model_agreement.csv"
    if not marker.exists():
        return
    required = [
        "results/tables/conversion_model_agreement.csv",
        "results/tables/conversion_efficiency_uncertainty.csv",
        "results/tables/conversion_profile_stability.csv",
        "results/tables/leave_group_out_conversion_stability.csv",
        "results/tables/outlier_conversion_sensitivity.csv",
        "results/tables/ghs_2019_integration_audit.csv",
        "results/tables/secondary_outcome_inventory.csv",
        "results/tables/conversion_monte_carlo_summary.csv",
        "results/tables/rf_benchmark_audit.csv",
        "paper_assets/tables/table6_robustness_summary.csv",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing

