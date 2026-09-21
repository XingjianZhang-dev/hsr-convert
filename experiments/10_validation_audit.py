from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.validation_audit import run_validation_audit


if __name__ == "__main__":
    run_validation_audit(ROOT)
