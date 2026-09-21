# Data Sources

This file records planned and implemented sources for SV-HSRDSS. The pipeline also writes a run-specific audit to `results/logs/source_audit.md`.

## World Bank WDI/HNP

- Source name: World Bank World Development Indicators and Health Nutrition and Population indicators.
- Access method: World Bank Indicators API, `https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?format=json&date=2015:2019`.
- Variables downloaded: configured in `config/indicators.yml`.
- Years covered in this pipeline: 2015-2019 only.
- License/usage note: World Bank Open Data terms should be reviewed before publication.
- Programmatic download: yes, via `src/download_worldbank.py`.

## Our World in Data Excess Mortality

- Source name: Our World in Data grapher dataset, cumulative excess deaths per million.
- Exact URL: `https://ourworldindata.org/grapher/cumulative-excess-deaths-per-million-covid.csv`.
- Variables downloaded: country/entity, ISO code, date, cumulative projected excess deaths per million.
- Years used in this pipeline: 2020-2022 only.
- License/usage note: Our World in Data generally publishes grapher data under CC BY unless otherwise stated on the dataset page; verify the current metadata before publication.
- Programmatic download: yes, via `src/download_owid.py`.

## WHO Health-System Resilience Indicator Package

- Source name: WHO health-system resilience and essential public health functions indicator materials.
- Access method: conceptual framework/manual review.
- Variables downloaded: none in the minimal pipeline.
- Usage in this pipeline: informs indicator dimensions only.
- Programmatic download: no.
- Manual note: identify and archive the exact WHO document/version before manuscript drafting.

## GHS Index

- Source name: Global Health Security Index.
- Access method: manual download or official data portal, depending on current availability.
- Variables downloaded: none in the minimal pipeline.
- Intended use: 2019 scores may be used as pre-shock preparedness indicators or benchmarks if a legitimate local file is provided. 2021 scores may only be comparator/sensitivity variables, not pre-shock features.
- Programmatic download: no in the minimal pipeline.
- Manual note: add file under `data/raw/ghs/` only after license and provenance are recorded.

## Stage 2 Expanded World Bank Blocks

- Source name: World Bank World Development Indicators and Health Nutrition and Population indicators.
- Access method: World Bank Indicators API, `https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?format=json&date=2015:2019`.
- Variables downloaded: configured in `config/indicator_blocks.yml`.
- Years covered: 2015-2019 only.
- Programmatic download: yes, via `src/feature_blocks.py`.
- Usage: pre-shock capacity, preparedness-proxy, vulnerability, equity/access, and context-control variables for HSR-Convert.

## Stage 3 Secondary Outcomes

- Source name: user-provided country-level secondary outcomes.
- Access method: local files under `data/raw/who_pulse/`, `data/raw/life_expectancy/`, or `data/raw/secondary_outcomes/`.
- Programmatic download: no.
- Usage: validation-only sensitivity outcomes.
- Status: skipped unless legitimate local CSV files with ISO3 country codes are provided.

## Stage 3 GHS 2019

- Source name: Global Health Security Index 2019.
- Access method: local file `data/raw/ghs/ghs_index_2019.csv`.
- Programmatic download: no.
- Usage: optional preparedness-block comparison and benchmark.
- Status: skipped unless a legitimate local file is provided; file hash and matched ISO3 counts are audited when present.

## Stage 4 World Bank Life Expectancy

- Source name: World Bank life expectancy at birth, total.
- Indicator: `SP.DYN.LE00.IN`.
- Access method: World Bank Indicators API, `https://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN?format=json&date=2019:2022`.
- Years used: 2019-2022.
- Programmatic download: yes, via `src/life_expectancy_outcome.py`.
- Usage: validation-only life-expectancy-loss outcomes after potential-capacity scoring.
- Output file: `data/processed/life_expectancy_loss_outcomes.csv`.
- License/usage note: World Bank Open Data terms should be reviewed before publication.

## Stage 4 GHS 2019

- Source name: Global Health Security Index 2019.
- Access method: official GHS raw CSV downloaded to `data/raw/ghs/ghs_index_2019.csv`.
- Exact URL: `https://ghsindex.org/wp-content/uploads/2022/04/2021-GHS-Index-April-2022.csv`.
- File hash SHA-256: `17197a2e8d19646e5ae9453ddbc3969fb084dbbb654795f00e14e84b45c71661`.
- Rows in official raw file: 390, covering 195 countries/territories for 2019 and 2021.
- Rows used in HSR-Convert: 2019 rows only, matched to 189 of 193 project scoring countries by country name/ISO3 mapping.
- Programmatic download: no repository downloader; downloaded as an official raw CSV and parsed locally by `src/load_ghs.py`.
- Usage: optional pre-shock preparedness correction and benchmark evidence variant.
- Provenance audit: `results/tables/stage4_ghs_2019_integration_audit.csv`.
- Status: loaded for Stage 4. No 2021 GHS values are used as pre-shock features.

## WHO Pulse Survey on Continuity of Essential Health Services

- Source name: WHO pulse survey on continuity of essential health services.
- Access method: manual download if public machine-readable data are available.
- Variables downloaded: none in the minimal pipeline.
- Intended use: secondary shock-period outcome only.
- Programmatic download: no in the minimal pipeline.

## Stage 4 WHO Pulse Survey

- Source name: WHO pulse survey on continuity of essential health services.
- Access method: local country-level CSV/XLSX/XLS files under `data/raw/who_pulse/` if supplied.
- Programmatic download: no.
- Usage: optional validation-only service-disruption outcome.
- Provenance audit: `results/tables/stage4_who_pulse_inventory.csv`.
- Status: skipped when no legitimate country-level machine-readable file is present.

## IHME/GBD or HAQ

- Source name: IHME Global Burden of Disease / HAQ Index.
- Access method: manual or API access if available under acceptable terms.
- Variables downloaded: none in the minimal pipeline.
- Intended use: optional benchmark or baseline health-status indicator.
- Programmatic download: no in the minimal pipeline.

## OECD Health Statistics

- Source name: OECD Health Statistics.
- Access method: optional high-income subset analysis, subject to access terms.
- Variables downloaded: none in the minimal pipeline.
- Programmatic download: no in the minimal pipeline.
