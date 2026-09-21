from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conversion_robustness import run_conversion_robustness


if __name__ == "__main__":
    run_conversion_robustness(ROOT, n_boot=1000)
