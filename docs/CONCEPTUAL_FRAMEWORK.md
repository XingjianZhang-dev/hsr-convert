# HSR-Convert Conceptual Framework

**HSR-Convert: Health-System Capacity-to-Resilience Conversion Framework** is Stage 2 of SV-HSRDSS. It reframes health-system resilience as a conversion problem rather than a direct country-ranking problem.

## 1. Potential Capacity

Potential Capacity is the uncertainty-aware pre-shock capacity score for country `i`:

```text
C_i = uncertainty-aware pre-shock capacity score
```

Higher `C_i` means stronger pre-shock potential capacity. It is built from 2015-2019 indicators only.

## 2. Shock Burden

Observed Shock Burden is the validation-only shock-period outcome:

```text
Y_i = observed shock-period burden
```

In the current pipeline, `Y_i` is cumulative excess deaths per million from 2020-2022. Higher `Y_i` means higher shock burden.

## 3. Expected Shock Burden

Expected Shock Burden estimates the burden expected from potential capacity and contextual constraints:

```text
Yhat_i = f(C_i, GDP_i, age_i, density_i, vulnerability_i, region_i)
```

This is a predictive association model, not a causal model.

## 4. Realized Resilience Gap

The realized gap is:

```text
G_i = Y_i - Yhat_i
```

Positive `G_i` means worse-than-expected realized outcome. Negative `G_i` means better-than-expected realized outcome.

## 5. Conversion Efficiency

Conversion Efficiency is oriented so higher is better:

```text
E_i = -standardized(G_i)
```

The pipeline also reports a percentile version:

```text
E_i = expected_burden_percentile_i - observed_burden_percentile_i
```

Higher `E_i` means stronger capacity-to-resilience conversion.

## 6. Conversion Profiles

Rule-based conversion profiles summarize the relationship between potential capacity, observed burden, and conversion efficiency:

- Effective converters
- Capacity under-realizers
- Adaptive over-performers
- Structurally vulnerable systems
- Uncertain / data-limited systems

These labels support diagnosis and comparison. They are not causal attributions.

## 7. Conversion Bottlenecks

Conversion bottlenecks are weak feature blocks relative to peer groups, such as same-income, same-region, and same-capacity-quartile countries. Bottleneck outputs identify likely diagnostic areas: capacity, preparedness proxies, vulnerability burden, and equity/access/financial protection.

## 8. Difference from Ordinary MCDM Ranking

Ordinary MCDM studies often produce point rankings and clusters. HSR-Convert first measures potential capacity under methodological uncertainty, then tests how that potential is converted into lower-than-expected shock burden. The main output is not just a rank; it is a conversion gap and bottleneck diagnosis.

## 9. Difference from Criticizing Existing Indicators

HSR-Convert does not claim that GHS, UHC, World Bank, or WHO indicators are useless. It treats them as capacity and context signals. The contribution is to show that capacity indicators need a shock-calibrated conversion layer before they can support resilience diagnosis.

## 10. Reviewer-Proof Evidence Package

Stage 3 adds robustness evidence around the same conceptual model. It asks whether conversion efficiency and profiles are stable across expected-burden models, bootstrap resamples, Monte Carlo perturbations, outcome windows, and country subsets.

Stage 3 also audits optional evidence sources. GHS 2019 and secondary outcomes are included only when legitimate local files exist. Random forest is treated as a sensitivity benchmark and is audited for leakage and overfitting before interpretation.

The framework should be advanced to manuscript drafting only if the evidence package can distinguish stable conversion diagnostics from unstable country labels.

## 11. Final Cross-Evidence Strengthening

Stage 4 keeps the same HSR-Convert framework and adds only final evidence checks. Life expectancy loss from World Bank `SP.DYN.LE00.IN` is used as a programmatic secondary validation outcome. Official GHS 2019 raw data are used as a provenance-controlled preparedness sensitivity, with 2021 values excluded from pre-shock scoring. WHO pulse survey evidence is used only when a legitimate country-level machine-readable file exists and is otherwise recorded as skipped.

The final diagnostic question is whether country conversion profiles, under-realizer and over-performer labels, and bottleneck patterns remain stable across the available evidence variants. Stable labels can be carried into Results writing as diagnostic evidence. Evidence-dependent and data-limited labels must be flagged as uncertain and should not be written as firm country classifications.

The large data-limited group is not a defect to hide. It is a design feature: HSR-Convert distinguishes stable diagnostics from cases where the country-level evidence base is too thin or outcome-dependent for firm labeling.

The manuscript contribution should be framed as capacity-to-resilience conversion for health-system resilience decision support. MCDM and machine-learning components are supporting mechanisms inside that architecture, not the central novelty by themselves.
