from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.final_paper_assets import refresh_final_paper_assets


if __name__ == "__main__":
    refresh_final_paper_assets(ROOT)
