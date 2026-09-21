# Data sources and licenses

This repository does not claim ownership over third-party data. Extracts are included only where the provider's terms permit redistribution, with attribution; everything else is fetched by `scripts/fetch_data.py` and verified against `data/EXPECTED_HASHES.json`.

## World Bank (redistributed)

World Bank indicator extracts under `data/raw/worldbank/` were pulled from the World Bank Indicators API on 2026-06-24 (see `results/logs/worldbank_download_records.json` and `results/logs/expanded_worldbank_download_records.json`). World Bank Open Data datasets are provided under the Creative Commons Attribution 4.0 International license (CC BY 4.0) unless otherwise stated. Attribution: The World Bank, World Development Indicators / Health Nutrition and Population Statistics.

- https://data.worldbank.org/summary-terms-of-use
- https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets

## Our World in Data (redistributed)

`data/raw/owid/cumulative-excess-deaths-per-million-covid.csv` is the OWID grapher CSV for cumulative excess deaths per million (based on The Economist's excess-mortality model), downloaded 2026-06-24. OWID data, charts, and articles are licensed under Creative Commons BY unless otherwise stated. Attribution: Our World in Data, "Cumulative excess deaths per million people" (Karlinsky and Kobak; The Economist).

- https://ourworldindata.org/faqs
- https://ourworldindata.org/easier-to-reuse-our-data

## Global Health Security Index (not redistributed)

The Stage 4 preparedness evidence uses the official GHS Index raw CSV (2019 and 2021 scores) from the GHS Index Report & Data page. The provider does not state a redistribution license for the raw file, so it is not included here. `python scripts/fetch_data.py` downloads it from the official URL to `data/raw/ghs/ghs_index_2019.csv` and checks its SHA-256 (`17197a2e8d19646e5ae9453ddbc3969fb084dbbb654795f00e14e84b45c71661`, 410,035 bytes). The pipeline uses the 2019 rows only.

- https://ghsindex.org/

## WHO pulse surveys (not ingested)

WHO pulse survey reports are cited as context in the manuscript. No country-level machine-readable file is used because none was legitimately available for reproducible merging (`data/raw/who_pulse/README.md`, `results/tables/stage4_who_pulse_inventory.csv`).
