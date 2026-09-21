# Secondary Outcome Manual Data Folder

Place legitimate country-level secondary outcome CSV files here. The pipeline will not fabricate values.

Required columns:
- `iso3`: ISO-3 country code.
- one or more numeric outcome columns.

Recommended metadata columns:
- `country`
- `source`
- `years`
- `direction`, with `higher_worse` or `higher_better`

Shock-period or post-shock outcomes are validation-only and must never be used as pre-shock features.
