from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.load_ghs import integrate_ghs_2019_stage4
from src.life_expectancy_outcome import run_life_expectancy_loss_validation
from src.who_pulse_outcome import run_who_pulse_optional_validation
from src.cross_evidence_stability import run_cross_evidence_stability
from src.stable_case_audit import run_stable_case_audit
from src.final_paper_assets import refresh_final_paper_assets
from src.stage4_summary import write_stage4_summary
from src.utils import audit_result_files, ensure_dirs


def main() -> None:
    ensure_dirs(ROOT)
    integrate_ghs_2019_stage4(ROOT)
    run_life_expectancy_loss_validation(ROOT)
    run_who_pulse_optional_validation(ROOT)
    run_cross_evidence_stability(ROOT)
    run_stable_case_audit(ROOT)
    write_stage4_summary(ROOT)
    refresh_final_paper_assets(ROOT)
    write_stage4_summary(ROOT)
    audit_result_files(ROOT)


if __name__ == "__main__":
    main()
