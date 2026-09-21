from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.life_expectancy_outcome import run_life_expectancy_loss_validation


if __name__ == "__main__":
    run_life_expectancy_loss_validation(ROOT)
