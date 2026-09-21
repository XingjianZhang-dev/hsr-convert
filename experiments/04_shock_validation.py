from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shock_validation import run_shock_validation


if __name__ == "__main__":
    run_shock_validation(ROOT)
