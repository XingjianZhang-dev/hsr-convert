# HSR-Convert

Code, configuration, processed evidence, and run summaries for the manuscript

**Capacity Is Not Resilience: An Uncertainty-Aware Decision-Support Framework for Health-System Capacity-to-Resilience Conversion Under Pandemic Shock** (Xingjian Zhang, Carnegie Mellon University; submitted to *Array*).

HSR-Convert links leakage-safe 2015–2019 country-level health-system indicators to 2020–2022 pandemic outcomes, estimates expected shock burden, converts residual performance into diagnostic profiles, and reports how much support each country label carries (stable, evidence-dependent, or data-limited). Everything in the paper is regenerated from this repository and its archived data snapshot.

## Contents

| Path | What it holds |
|---|---|
| `src/` | pipeline modules (feature blocks, expected-burden models, conversion, Monte Carlo, cross-evidence stability) |
| `experiments/` | staged entry points (`run_minimal_pipeline.py`, `run_stage2_hsr_convert.py`, `run_stage3_reviewer_proof.py`, `run_stage4_final_evidence.py`) |
| `revision_analysis/` | sensitivity analyses (weights, thresholds, Monte Carlo settings), repeated-CV model comparison, static-index baselines, permutation/bootstrap/calibration statistics, and figures |
| `config/` | indicator, feature-block, outcome, and ensemble-grid configuration |
| `data/raw/` | archived World Bank and Our World in Data extracts (snapshot of 2026-06-24); the GHS raw file is fetched by script |
| `data/processed/` | processed pre-shock indicator matrices and outcome tables |
| `results/tables/`, `results/figures/` | pipeline outputs (Stages 1–4) |
| `results/revision/` | outputs of the revision analyses, including `revision_numbers.json`, the single source of every number in the manuscript |
| `paper_assets/` | manuscript-facing tables and figures |
| `scripts/` | `fetch_data.py` (download + SHA-256 verification), `verify_claims.py`, `round3_publication_assets.py` (figures/tables) |
| `tests/` | regression, leakage-safety, and output tests |
| `docs/` | data card, data sources, methods notes, conceptual framework, run summary |

## Setup

```bash
git clone https://github.com/XingjianZhang-dev/hsr-convert.git
cd hsr-convert
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements-lock.txt for the exact environment
python scripts/fetch_data.py             # downloads the GHS raw file and verifies its SHA-256
```

Python 3.12 was used; the pinned versions in `requirements.txt` are the ones behind the reported numbers. No GPU is needed.

## One-command reproduction

```bash
make verify      # seconds: shipped tables carry the manuscript numbers
make test        # ~30 s: 41 tests (leakage guards, score direction, output shapes)
make smoke       # ~1 min: verify + tests + Stage 4 rerun + threshold-sensitivity grid
make reproduce   # ~5 min: Stages 1-4 from the archived raw extracts (HSR_OFFLINE=1)
make revision    # ~15 min on 8 cores: all revision analyses -> results/revision/
make figures     # manuscript figures and LaTeX tables from the result tables
make all         # reproduce + revision + figures + verify
```

`HSR_OFFLINE=1` (the Makefile default) makes the download modules reuse the archived extracts under `data/raw/`; unset it, or run `python scripts/fetch_data.py --refresh-open-data`, to re-download from the World Bank and OWID APIs (upstream revisions can change values, in which case the result is a data update rather than a reproduction).

## What each stage produces

| Stage | Command | Main outputs |
|---|---|---|
| 1 | `experiments/run_minimal_pipeline.py` | indicator matrix, MCDM ensemble ranks, shock-validation correlations |
| 2 | `experiments/run_stage2_hsr_convert.py` | 26-indicator feature blocks, potential-capacity scores (n=193), expected-burden models, conversion profiles (n=106), bottlenecks, ablations |
| 3 | `experiments/run_stage3_reviewer_proof.py` | bootstrap and Monte Carlo uncertainty, model agreement, RF audit, paper tables 1–6 |
| 4 | `experiments/run_stage4_final_evidence.py` | GHS-enhanced variant, life-expectancy-loss variant, cross-evidence agreement, label triage, stable-case audit, paper tables 7–8 |
| R | `python -m revision_analysis.run_all` | `results/revision/*.csv`, `stat_tests.json`, `revision_numbers.json`, figures 6–7 and supplementary figures 6–7 |

Headline numbers (also checked by `make verify`): 193 scoring countries, 106 validation countries, 189 GHS-matched countries; 41 stable / 38 evidence-dependent / 107 data-limited labels; OWID-baseline vs GHS-enhanced profile agreement 0.972 (Spearman 0.996); vs life-expectancy loss 0.642 (0.660); 89 of 106 labels invariant across the 25-combination threshold grid.

## Design notes

* Predictors are 2015–2019 only; 2020–2022 variables are outcomes. GHS 2019 is allowed as a predictor, GHS 2021 is not (`tests/test_no_data_leakage.py`).
* The baseline scoring frame uses World Bank preparedness proxies only (`config/indicator_blocks.yml: baseline_blocks.include_ghs_2019: false`); GHS 2019 enters through the Stage 4 GHS-enhanced variant.
* One ridge preprocessor (`src/conversion_robustness.numeric_scaler`) is shared by every refit path, so cross-evidence disagreement reflects the evidence and not the estimator.
* Random seeds: 20260624 (pipeline), 20260921 (revision analyses).

## Data and licenses

See `DATA_LICENSES.md` and `docs/DATA_SOURCES.md`. World Bank and OWID extracts are redistributed under CC BY 4.0 with attribution to the providers; the GHS Index raw file is not redistributed and is fetched from the official URL with hash verification; no WHO pulse country-level file is used.

## Interpretation boundary

All outputs are country-level, ecological, descriptive decision-support diagnostics. They are not clinical advice, emergency triage, causal policy estimates, or country accountability scores.

## License and citation

Code and generated outputs: MIT (see `LICENSE`). Please cite the manuscript and this repository (`CITATION.cff`).
