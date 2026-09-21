from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.block_ablation import run_block_ablation_and_benchmarks
from src.conversion_bottlenecks import run_conversion_bottlenecks
from src.feature_blocks import build_expanded_indicator_matrix
from src.resilience_conversion import run_resilience_conversion
from src.stage2_summary import write_hsr_convert_summary
from src.utils import audit_result_files, ensure_dirs
from src.validation_audit import run_validation_audit


def main() -> None:
    ensure_dirs(ROOT)
    # Initial audit uses the existing minimal outputs before feature expansion.
    run_validation_audit(ROOT)
    build_expanded_indicator_matrix(ROOT)
    # Rerun the audit so final diagnostics include expanded features and population groups.
    run_validation_audit(ROOT)
    run_resilience_conversion(ROOT)
    run_conversion_bottlenecks(ROOT)
    run_block_ablation_and_benchmarks(ROOT)
    write_hsr_convert_summary(ROOT)
    audit_result_files(ROOT)


if __name__ == "__main__":
    main()
