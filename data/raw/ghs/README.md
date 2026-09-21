# GHS Index 2019 source note

The pipeline uses the official GHS Index raw CSV as pre-shock preparedness evidence (2019 rows only). The file is not redistributed in this repository because the provider does not state redistribution terms. Fetch it with

```bash
python scripts/fetch_data.py
```

which downloads `https://ghsindex.org/wp-content/uploads/2022/04/2021-GHS-Index-April-2022.csv` to `data/raw/ghs/ghs_index_2019.csv` and verifies its SHA-256 against `data/EXPECTED_HASHES.json`.

Accepted formats for a manually placed file:

1. The official raw CSV containing `Country`, `Year`, `OVERALL SCORE`, and the six category score columns; the loader filters `Year == 2019` and maps country names to ISO3 codes using the World Bank country sample.
2. A pre-standardized CSV with `iso3` and `ghs_overall_2019` (optionally `ghs_prevention_2019`, `ghs_detection_2019`, `ghs_response_2019`, `ghs_health_system_2019`, `ghs_compliance_2019`, `ghs_risk_environment_2019`).

If the file is absent, Stage 4 records `skipped_missing_file` and no GHS values are fabricated.
