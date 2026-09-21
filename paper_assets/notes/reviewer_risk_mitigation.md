# Reviewer Risk Mitigation

## Preparedness and Service-Disruption Evidence

- GHS 2019 is loaded from the audited official raw CSV with 189 matched scoring countries.
- WHO pulse survey remains skipped because no legitimate country-level machine-readable file is present. This is a limitation of service-disruption validation, not evidence of no disruption.

## Label Uncertainty

- Stable labels: 41; evidence-dependent labels: 38; data-limited labels: 107.
- The large data-limited group should be framed as a diagnostic safeguard: the system refuses to over-classify countries with insufficient cross-evidence support.

## Ecological Analysis Boundary

- All outputs are country-level diagnostic associations. They are not individual-level, clinical, emergency-response, or causal policy estimates.
- Life expectancy loss is a secondary outcome and may reflect broader health-system, demographic, and reporting factors beyond COVID-period mortality.

## Method Contribution

- The core contribution is capacity-to-resilience conversion for health-system resilience decision support.
- The paper should emphasize a problem-specific decision-support architecture: pre-shock capacity scoring, expected shock-burden modeling, realized resilience gaps, conversion efficiency, profile stability, and bottleneck diagnostics.
- MCDM and ML components are supporting modules inside HSR-Convert, not the claimed contribution by themselves.
