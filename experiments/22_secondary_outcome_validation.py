from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.secondary_outcomes import run_secondary_outcome_validation


if __name__ == "__main__":
    run_secondary_outcome_validation(ROOT)
