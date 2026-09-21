from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_blocks import build_expanded_indicator_matrix


if __name__ == "__main__":
    build_expanded_indicator_matrix(ROOT)
