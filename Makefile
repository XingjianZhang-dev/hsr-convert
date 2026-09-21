# HSR-Convert reproduction targets. All steps run on CPU; HSR_OFFLINE=1 reuses the archived data snapshot.
PYTHON ?= python
export HSR_OFFLINE ?= 1

.PHONY: env fetch verify test smoke reproduce revision all clean

env:
	$(PYTHON) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

fetch:          ## download the GHS raw file (not redistributed) and verify SHA-256 hashes
	$(PYTHON) scripts/fetch_data.py

verify:         ## check that the shipped tables carry the manuscript numbers (seconds)
	$(PYTHON) scripts/verify_claims.py

test:           ## regression, leakage and output tests (about 30 s)
	$(PYTHON) -m pytest -q

smoke:          ## reduced end-to-end check: Stage 4 rerun + tests + verification + threshold grid (about 1 min)
	$(PYTHON) experiments/run_stage4_final_evidence.py
	$(PYTHON) -m pytest -q
	$(PYTHON) scripts/verify_claims.py
	$(PYTHON) -m revision_analysis.smoke

reproduce:      ## full pipeline from the archived raw extracts (about 5 min): stages 1-4
	$(PYTHON) experiments/run_minimal_pipeline.py
	$(PYTHON) experiments/run_stage2_hsr_convert.py
	$(PYTHON) experiments/run_stage3_reviewer_proof.py
	$(PYTHON) experiments/run_stage4_final_evidence.py

revision:       ## sensitivity, model comparison, baselines, statistics and figures (about 15 min on 8 cores)
	$(PYTHON) -m revision_analysis.run_all

figures:        ## manuscript figures and LaTeX tables from the result tables
	$(PYTHON) scripts/round3_publication_assets.py

all: reproduce revision figures verify

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
