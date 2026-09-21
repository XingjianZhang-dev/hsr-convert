from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conversion_robustness import run_conversion_robustness
from src.load_ghs import integrate_ghs_2019
from src.secondary_outcomes import run_secondary_outcome_validation
from src.conversion_monte_carlo import run_conversion_monte_carlo
from src.rf_benchmark_audit import run_rf_benchmark_audit
from src.paper_assets import build_paper_assets
from src.stage3_summary import write_stage3_summary
from src.utils import audit_result_files, ensure_dirs


def main() -> None:
    ensure_dirs(ROOT)
    run_conversion_robustness(ROOT, n_boot=1000)
    integrate_ghs_2019(ROOT)
    run_secondary_outcome_validation(ROOT)
    run_conversion_monte_carlo(ROOT, n_sim=1000)
    run_rf_benchmark_audit(ROOT)
    write_stage3_summary(ROOT)
    build_paper_assets(ROOT)
    write_stage3_summary(ROOT)
    audit_result_files(ROOT)


if __name__ == "__main__":
    main()
