"""Run every revision analysis in order (A2 before A4, because A4 reuses A2's out-of-fold predictions)."""

from __future__ import annotations

import time

from revision_analysis import a0_pipeline_numbers, a1_sensitivity, a2_model_comparison, a3_baselines, a4_statistics, a5_figures


def main() -> None:
    for name, fn in [
        ("A0 pipeline numbers", a0_pipeline_numbers.run),
        ("A2 model comparison", a2_model_comparison.run),
        ("A1 sensitivity", a1_sensitivity.run),
        ("A3 baselines", a3_baselines.run),
        ("A4 statistics", a4_statistics.run),
        ("A5 figures", a5_figures.run),
    ]:
        t0 = time.time()
        fn()
        print(f"[done] {name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
