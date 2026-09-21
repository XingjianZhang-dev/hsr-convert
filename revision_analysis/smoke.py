"""Fast smoke test of the revision analyses on the shipped tables (threshold grid + headline numbers)."""

from __future__ import annotations

from revision_analysis import a0_pipeline_numbers, a1_sensitivity


def main() -> None:
    numbers = a0_pipeline_numbers.run()
    _, _, summary = a1_sensitivity.threshold_sensitivity()
    triage = numbers["label_triage"]
    print(f"label triage: stable={triage['stable']} evidence-dependent={triage['evidence_dependent']} data-limited={triage['data_limited']}")
    print(f"threshold grid: {summary['n_countries_invariant_main_grid']}/{summary['n_countries']} countries invariant across {summary['n_combinations_main_grid']} combinations")
    assert triage["stable"] + triage["evidence_dependent"] + triage["data_limited"] == triage["n_labelled"]
    print(f"{triage['n_scoring_without_any_lens']} scoring countries have no outcome lens (missing model covariates) and receive no label")
    print("smoke test passed")


if __name__ == "__main__":
    main()
