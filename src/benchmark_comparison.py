from __future__ import annotations

from pathlib import Path

from .block_ablation import run_benchmark_comparison
from .utils import ROOT


if __name__ == "__main__":
    run_benchmark_comparison(Path(ROOT))
