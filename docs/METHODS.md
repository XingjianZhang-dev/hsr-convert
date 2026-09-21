# Methods

This document describes the implemented minimal pipeline. It is intentionally limited to methods already executed by the repository scripts.

## HSR-Convert Framework

Stage 2 implements **HSR-Convert: Health-System Capacity-to-Resilience Conversion Framework**. The framework treats pre-shock health-system indicators as potential capacity, not realized resilience. Shock-period outcomes are introduced only after potential capacity scores are generated.

For country `i`:

- Potential Capacity: `C_i`, an uncertainty-aware pre-shock potential-capacity score.
- Observed Shock Burden: `Y_i`, cumulative excess deaths per million or another validation-only shock outcome.
- Expected Shock Burden: `Yhat_i = f(C_i, GDP_i, age_i, density_i, vulnerability_i, region_i, income_i)`.
- Realized Resilience Gap: `G_i = Y_i - Yhat_i`.
- Conversion Efficiency: `E_i = -standardized(G_i)` and a percentile alternative `expected_burden_percentile_i - observed_burden_percentile_i`.

Higher `C_i` means stronger pre-shock potential. Higher `Y_i` means higher shock burden. Positive `G_i` means worse-than-expected realized burden. Higher `E_i` means stronger capacity-to-resilience conversion. These are predictive associations and diagnostic comparisons, not causal estimates.

## Feature Blocks

The expanded Stage 2 matrix groups 2015-2019 World Bank indicators into:

- Capacity: workforce, beds, and health-financing resources.
- Preparedness proxies: immunization and infectious-disease control proxies available before 2020; GHS 2019 is documented as a manual source and not fabricated.
- Vulnerability: demographic, baseline-health, air-pollution, and mortality-burden indicators oriented so higher block scores mean lower vulnerability burden.
- Equity/access/financial protection: UHC, out-of-pocket burden, water, sanitation, and handwashing access.

The original minimal feature matrix is retained for comparison.

## Feature Construction

Input indicators are drawn from World Bank WDI/HNP data for 2015-2019. For each country and indicator, the pipeline computes the 2015-2019 mean when at least three annual observations are available. Otherwise, it uses the latest available pre-2020 value and records the number of observed years.

Only indicators marked `use_as_feature: true` in `config/indicators.yml` are used in resilience scoring. Context controls such as GDP per capita are retained for validation models but excluded from the MCDM score.

## Imputation

The ensemble currently supports median imputation, KNN imputation, and iterative imputation. Imputation is applied after selecting the primary sample and before normalization.

## Normalization

All normalizers orient indicators so that larger values represent more resilient conditions. Cost indicators are reversed before scaling. Implemented normalizers are min-max, robust min-max, vector, and z-score-to-min-max.

## Weighting

Implemented weighting schemes are equal weights, standard-deviation weights, entropy weights, CRITIC weights, and MEREC-style leave-one-criterion weights. All output weights are nonnegative and sum to one.

## Ranking

Implemented rankers are TOPSIS, VIKOR, MARCOS-style utility, EDAS, CODAS, WASPAS, and MABAC. Scores are oriented so that larger score means higher resilience; ranks are ascending from 1 as the highest-resilience country.

## Methodological Ensemble

The ensemble evaluates the Cartesian product of configured imputation, normalization, weighting, and ranking methods. It saves per-pipeline ranks and consensus rank distributions, including median rank, mean rank, rank standard deviation, 5-95 percent rank interval, top-quartile probability, bottom-quartile probability, and method disagreement metrics.

## Shock Validation

The validation outcome is cumulative excess deaths per million from Our World in Data, restricted to observations dated from 2020-01-01 through 2022-12-31. Outcomes are never merged into the feature matrix before scoring.

Validation reports Spearman and Kendall correlations between pre-shock resilience outputs and later shock outcomes, plus bootstrap confidence intervals. Regression-style checks use standardized covariates and linear models with bootstrap confidence intervals, controlling for available pre-shock GDP per capita, age structure, population density, income group, and region.

The appropriate interpretation is external consistency or predictive association, not causality.

## Validation Audit

Stage 2 audits score direction before making conversion claims. Consensus score, median rank, mean rank, top-quartile probability, bottom-quartile probability, and rank instability are correlated with shock outcomes using Spearman and Kendall statistics with bootstrap confidence intervals. Outcome windows are recomputed through 2020, 2021, and 2022. The audit also reports validation-sample bias, outlier and microstate sensitivity where population is available, and indicator/dimension associations with outcomes.

## Expected Shock Burden Model

Expected burden models use only post-scoring validation data. Implemented variants are linear regression, ridge regression, Huber robust regression, and random forest sensitivity when sample size supports it. Cross-validated RMSE is reported. Ridge is used as the primary conversion model because it is stable, interpretable, and CPU-friendly.

## Conversion Profiles

Rule-based profiles are assigned from potential capacity, observed burden, conversion efficiency, rank uncertainty, and missingness:

- Effective converters
- Capacity under-realizers
- Adaptive over-performers
- Structurally vulnerable systems
- Uncertain / data-limited systems

Profiles are diagnostic labels, not causal claims or prescriptive medical advice.

## Bottleneck Analysis

Bottleneck diagnostics compare country block scores with same-income, same-region, and same-capacity-quartile peers. Negative standardized block gaps identify likely conversion bottleneck areas such as preparedness proxies, vulnerability burden, equity/access/financial protection, or general capacity.

## Block Ablation and Benchmarks

Block ablations compare original minimal scoring, single blocks, selected block combinations, and all blocks. Benchmarks include UHC coverage, health expenditure, physicians per 1,000, equal-weight and PCA composites, selected MCDM pipelines, ridge prediction, and random forest sensitivity when sample size is sufficient. The benchmark goal is to test whether HSR-Convert adds uncertainty-aware capacity scoring, shock-calibrated conversion efficiency, and bottleneck diagnosis, not to replace GHS, UHC, or WHO indicators.

## Stage 3 Conversion Robustness

Stage 3 compares conversion efficiency rankings across linear, ridge, Huber, and random-forest sensitivity models. It reports pairwise Spearman/Kendall agreement, top-10 under-realizer overlap, top-10 over-performer overlap, and profile assignment agreement.

Bootstrap conversion uncertainty resamples validation countries and refits the expected-burden model. Reported quantities include conversion efficiency intervals, under-realizer probability, over-performer probability, uncertain-profile probability, and profile stability.

Leave-group-out stability excludes regions, income groups, high-income/non-high-income subsets, microstates where population is available, and top/bottom outcome outliers.

## GHS Integration

GHS 2019 is loaded only from `data/raw/ghs/ghs_index_2019.csv` if a legitimate local file is supplied. Stage 4 accepts the official GHS raw CSV containing 2019 and 2021 rows, filters `Year == 2019`, maps country names to project ISO3 codes, and extracts the overall and six category scores. The loader records file hash, detected columns, matched ISO3 codes, unmatched countries, and missingness. If absent, the module writes skipped audit outputs and does not fabricate values. GHS 2021 values are never used as pre-shock features.

## Secondary Outcomes

Secondary outcomes are loaded only from local country-level CSV files under `data/raw/who_pulse/`, `data/raw/life_expectancy/`, or `data/raw/secondary_outcomes/`. If no legitimate file is present, the module writes inventory and skipped-validation outputs.

## Stage 4 Final Evidence Strengthening

Stage 4 is restricted to final evidence strengthening before manuscript drafting. It does not add additional MCDM methods, clustering, random-forest optimization, or causal interpretation.

GHS 2019 integration uses the official raw GHS CSV when present under `data/raw/ghs/ghs_index_2019.csv`. The Stage 4 loader records file path, SHA-256 hash, detected columns, row count, matched ISO3 countries, unmatched countries, category missingness, coverage, and source provenance. GHS is used to compare World Bank preparedness proxies, GHS-only preparedness, and a combined preparedness score, then to recompute conversion profiles as a sensitivity evidence variant.

Life expectancy loss is constructed programmatically from the World Bank `SP.DYN.LE00.IN` indicator for 2019-2022. The validation outcomes are `life_expectancy_loss_2019_2020`, `life_expectancy_loss_2019_2021`, `life_expectancy_loss_2019_2022`, and `max_life_expectancy_drop_2020_2022`, oriented so higher values mean worse shock-period health outcomes. These variables are merged only after potential capacity and block scores are already generated.

WHO pulse survey validation is optional. The pipeline inspects `data/raw/who_pulse/` for CSV/XLSX/XLS files with ISO3-like country codes and a numeric country-level disruption measure. If no legitimate file exists, it updates the folder README, writes skipped outputs, and does not infer disruption values from PDFs or figures.

Cross-evidence stability compares the baseline OWID excess-mortality conversion results with available Stage 4 variants. It reports conversion-efficiency rank association, profile agreement, under-realizer overlap, over-performer overlap, bottleneck agreement, and country label categories that distinguish stable diagnostics from evidence-dependent labels.

The stable case audit selects countries from stable under-realizer, stable over-performer, stable effective-converter, structurally vulnerable, evidence-dependent, and data-limited groups where available. Each row reports profile stability, conversion efficiency and interval, observed and expected burden, realized gap, block scores, bottleneck signal, missingness, validation outcomes, label stability, and a descriptive diagnostic note.

Evidence-dependent and data-limited labels are retained as uncertainty outputs. They should not be collapsed into stable under-realizer or over-performer claims. This is intentional: HSR-Convert is designed to identify when country-level evidence is insufficient for a firm diagnostic label.

All Stage 4 evidence remains country-level and ecological. It supports diagnostic external consistency and method evaluation, not causal policy inference, clinical advice, or emergency triage.

## Monte Carlo Conversion Uncertainty

The Monte Carlo module perturbs block weights, potential-capacity scores, method uncertainty, bounded score noise, optional block omission, and expected-burden bootstrap samples. It reports conversion intervals, profile probabilities, and bottleneck probabilities.

## Random Forest Audit

Random forest remains a sensitivity benchmark. Stage 3 audits exact RF features, screens them for outcome/post-shock leakage terms, runs repeated K-fold CV, leave-one-region-out and leave-one-income-group-out CV, a shuffled-outcome negative control, and permutation importance.

## Paper Asset Construction

`paper_assets/` contains tables, figures, and notes derived from result files. These are evidence assets for later manuscript drafting, not a manuscript.

Stage 4 refreshes final paper assets by rebuilding the Stage 3 tables and figures, then adding cross-evidence stability and stable case audit assets. The notes state claims that can and cannot be made from the diagnostic evidence.

For journal positioning, the contribution should be described as a problem-specific decision-support architecture for capacity-to-resilience conversion. MCDM and machine-learning components provide uncertainty-aware scoring and sensitivity evidence, but the contribution is not simply a larger collection of MCDM/ML methods.
