from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.conversion_bottlenecks import run_conversion_bottlenecks


if __name__ == "__main__":
    run_conversion_bottlenecks(ROOT)
