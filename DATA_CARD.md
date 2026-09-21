# Data Card

## Intended Use

The dataset supports a pre-shock health-system resilience scoring and validation pipeline. It is intended for research diagnostics and method comparison, not for clinical advice, emergency response triage, or causal policy claims.

Stage 2 HSR-Convert outputs are decision-support diagnostics. They identify potential capacity, expected shock burden, realized resilience gaps, conversion efficiency, and bottleneck patterns. They should not be interpreted as medical advice or as proof that changing one indicator would causally change shock outcomes.

## Input Window

Features are restricted to 2015-2019. The pipeline records observed-year counts per indicator and country and excludes any feature configured with years beyond 2019.

## Validation Window

Shock outcomes are restricted to 2020-2022 and are used only after resilience scores are generated.

## Known Limitations

- World Bank indicator coverage varies substantially by country and indicator.
- Missing data are imputed for scoring, so rankings should be interpreted through uncertainty intervals.
- Excess mortality coverage is incomplete and depends on mortality registration capacity and modeling choices.
- Mortality registration and excess-mortality modeling quality vary across countries and may be related to system capacity.
- GHS Index 2019 is not fabricated. Stage 4 uses the official raw GHS CSV placed under `data/raw/ghs/` and filters 2019 rows only.
- Stage 2 conversion efficiency is shock-calibrated and sample-dependent; it should be interpreted as an external-consistency diagnostic, not a universal country trait.
- Stage 3 robustness intervals are conditional on the available OWID validation sample and configured World Bank indicators.
- Optional secondary outcomes are absent unless legitimate local files are supplied; skipped outputs are explicit and should not be interpreted as evidence.
- Random-forest outputs are sensitivity checks and may reflect sample-specific nonlinear patterns.
- Stage 4 life-expectancy-loss outcomes are secondary validation outcomes from World Bank `SP.DYN.LE00.IN`; they can reflect demographic, registration, and broader health-system factors beyond COVID-period mortality.
- Stage 4 GHS evidence is provenance-controlled. The loaded official raw CSV is parsed for 2019 values only; 2021 GHS values are not used as pre-shock features.
- Stage 4 WHO pulse evidence is optional and requires a legitimate country-level machine-readable file. Absence of such a file is recorded as skipped, not as evidence of no service disruption.
- Cross-evidence country labels should be interpreted through label stability. Evidence-dependent and data-limited labels should not be treated as stable country classifications.
- Data-limited labels are a deliberate uncertainty output, not a failed classification. They indicate that HSR-Convert does not force country-specific conclusions when cross-evidence support is weak.

## Leakage Prevention

The test suite checks that configured feature indicators do not use years from 2020 onward and that common COVID response/outcome terms are absent from feature names and source codes.

Shock outcomes remain separate from pre-shock scoring. The conversion modules merge outcomes only after potential capacity scores and block scores have been generated.

Stage 3 RF and Monte Carlo tests include explicit leakage checks, but they remain country-level predictive association analyses and cannot support causal policy interpretation.

Stage 4 life expectancy and WHO pulse variables are merged only after potential-capacity scoring. They are validation outcomes or sensitivity checks, not pre-shock scoring features.

## Interpretation Guardrails

This is a country-level ecological analysis. Outputs are suitable for diagnostic comparison, external-consistency assessment, and method evaluation. They are not individual-level risk estimates, clinical advice, emergency-response triage, or causal policy effects.

The main methodological contribution is capacity-to-resilience conversion: separating pre-shock potential capacity from shock-realized outcomes through expected burden, realized resilience gaps, conversion efficiency, profile stability, and bottleneck diagnostics. MCDM and machine-learning components are supporting modules, not the central claim by themselves.
