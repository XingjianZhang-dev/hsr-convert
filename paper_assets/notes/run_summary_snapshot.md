# Run Summary

Generated: 2026-09-21 19:13:13 EDT
Git commit: 5a93542

## Stage 3 Reviewer-Proof Evidence Package

HSR-Convert measures how pre-shock health-system capacity is converted into shock-realized resilience. Stage 3 adds robustness, uncertainty, provenance, RF sensitivity auditing, and paper-ready evidence assets.

## Coverage

- Expanded scoring countries: 193
- Validation countries in conversion models: 106
- Usable expanded indicators: 26
- Feature blocks: capacity, equity_access, preparedness, vulnerability

## GHS and Secondary Outcomes

- GHS 2019 status: loaded
- Secondary outcome status: skipped_no_legitimate_country_level_file

## Robustness Findings

- Model agreement mean Spearman=0.818, minimum Spearman=0.669, mean profile agreement=0.893.
- Bootstrap median profile stability score: 1.000
- Monte Carlo robust-profile share: 0.802

Stable under-realizer candidates:
- Argentina (ARG): P(under-realizer)=1.000
- United Kingdom (GBR): P(under-realizer)=1.000
- Kazakhstan (KAZ): P(under-realizer)=1.000
- Colombia (COL): P(under-realizer)=1.000
- Chile (CHL): P(under-realizer)=1.000

Stable over-performer candidates:
- Namibia (NAM): P(over-performer)=1.000
- Mongolia (MNG): P(over-performer)=1.000
- Tajikistan (TJK): P(over-performer)=1.000
- Kyrgyz Republic (KGZ): P(over-performer)=1.000
- Barbados (BRB): P(over-performer)=1.000

Stable bottleneck patterns:
- capacity: mean bottleneck probability 0.811
- vulnerability: mean bottleneck probability 0.120
- equity_access: mean bottleneck probability 0.058
- preparedness: mean bottleneck probability 0.011

## Random Forest Audit

- RF sensitivity repeated K-fold RMSE=1846.5 (std 44.0), no leakage detected=True.
- RF remains a sensitivity benchmark, not the main HSR-Convert model.

## Paper Assets

- Paper tables generated: 8
- Paper figures generated: 15

## Final Evidence Statement

- A. HSR-Convert is robust enough to start manuscript drafting, with unstable country labels flagged.

## Skipped Datasets and Failed Experiments

- GHS 2019 integration is skipped unless a legitimate local file is provided.
- Secondary outcomes are skipped unless legitimate country-level files are provided.
- Optional profile clustering was not run because Stage 3 focused on hardening the rule-based HSR-Convert framework.

## Limitations

- Excess mortality coverage remains incomplete.
- Conversion models are predictive association models and do not support causal policy claims.
- Country labels with low stability should be described as uncertain.
- RF sensitivity must not be treated as the primary evidence.

## Next Recommended Experiment

Add legitimate GHS 2019 and WHO pulse survey country-level data, then rerun Stage 3 to test whether preparedness and service-disruption evidence changes conversion profiles.