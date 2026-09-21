from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_indicator_matrix import build_indicator_matrix
from src.pipeline_ensemble import run_baseline


if __name__ == "__main__":
    build_indicator_matrix(ROOT)
    run_baseline(ROOT)
