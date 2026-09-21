from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.resilience_conversion import run_resilience_conversion


if __name__ == "__main__":
    run_resilience_conversion(ROOT)
