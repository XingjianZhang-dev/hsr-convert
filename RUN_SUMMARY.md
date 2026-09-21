# Run Summary

Generated: 2026-09-21 19:13:13 EDT
Git commit: 5a93542

## Stage 4 Final Evidence Strengthening

Stage 4 adds only final high-value evidence before manuscript drafting: provenance-controlled GHS integration, programmatic life-expectancy-loss validation, optional WHO pulse validation, cross-evidence stability, stable case audits, and refreshed paper assets.

## Coverage

- Scoring countries: 193
- Conversion validation countries, OWID baseline: 106
- Life expectancy outcome max coverage: 193
- Usable expanded indicators: 26
- Feature blocks: capacity, equity_access, preparedness, vulnerability

## Source Status

- GHS status: loaded
- WHO pulse status: skipped_no_legitimate_file
- Life expectancy loss: downloaded programmatically from World Bank SP.DYN.LE00.IN for 2019-2022.

## Cross-Evidence Stability

- Mean profile agreement across evidence variants: 0.755
- Stable country labels: 41
- Evidence-dependent labels: 38
- Data-limited labels: 107

Interpretation guardrail: data-limited and evidence-dependent labels are retained as uncertainty outputs. They should be presented as a strength of HSR-Convert, because the system does not force stable country claims when cross-evidence support is weak.

Stable under-realizers:
- Belarus (BLR): E=-2.935, bottleneck=vulnerability
- Slovak Republic (SVK): E=-1.403, bottleneck=vulnerability
- United States (USA): E=-1.216, bottleneck=equity_access
- Croatia (HRV): E=-0.983, bottleneck=preparedness
- Romania (ROU): E=-0.935, bottleneck=preparedness

Stable over-performers:
- Dominican Republic (DOM): E=1.507, bottleneck=capacity
- Suriname (SUR): E=0.471, bottleneck=preparedness
- Bahamas, The (BHS): E=0.332, bottleneck=capacity

Stable bottleneck patterns:
- Mean bottleneck agreement across evidence variants: 0.660

## Paper Assets

- Final paper tables: 8
- Final paper figures: 15

## Final Evidence Statement

- A. ready for manuscript drafting, with evidence-dependent country labels flagged.

## Skipped Datasets and Why

- GHS 2019 was loaded from `data/raw/ghs/ghs_index_2019.csv`; coverage 189 countries; file hash 17197a2e8d19646e5ae9453ddbc3969fb084dbbb654795f00e14e84b45c71661.
- WHO pulse survey is skipped because no legitimate country-level CSV/XLSX/XLS file is present.

## Reviewer Risk Mitigation

- Preparedness evidence: GHS 2019 is handled as provenance-controlled preparedness sensitivity evidence; 2021 GHS values are not used as pre-shock features.
- Service-disruption evidence: WHO pulse survey validation remains optional and is skipped unless a legitimate country-level machine-readable file exists. This is a limitation, not a negative service-disruption finding.
- Label uncertainty: stable, evidence-dependent, and data-limited labels are separate outputs. Country-specific Results claims should prioritize stable labels and explicitly flag the other groups.
- Ecological boundary: all analyses are country-level diagnostic associations and must not be written as causal, clinical, emergency-response, or prescriptive policy conclusions.
- Method contribution: the paper should foreground capacity-to-resilience conversion rather than a generic new MCDM framework.

## Limitations

- Life expectancy loss is a secondary health outcome and may reflect broader demographic and reporting factors.
- Cross-evidence labels are diagnostic stability labels, not causal classifications.
- Bottleneck diagnostics are descriptive and should not be written as medical or policy prescriptions.

## Next Step

Start manuscript drafting. Rerun Stage 4 only if a legitimate WHO pulse country-level file or replacement GHS source is supplied before drafting.