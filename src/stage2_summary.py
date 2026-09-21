from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

from .utils import ROOT, now_stamp, write_text


def _git_hash(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not available"


def _fmt(value: object, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_hsr_convert_summary(root: Path = ROOT) -> Path:
    root = Path(root)
    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix_expanded.csv")
    meta = pd.read_csv(root / "data/processed/indicator_metadata_expanded.csv")
    block_missing = pd.read_csv(root / "results/tables/block_missingness.csv")
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    score_audit = pd.read_csv(root / "results/tables/score_rank_direction_audit.csv")
    window = pd.read_csv(root / "results/tables/outcome_window_sensitivity.csv")
    efficiency = pd.read_csv(root / "results/tables/conversion_efficiency_scores.csv")
    profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    bottlenecks = pd.read_csv(root / "results/tables/conversion_bottleneck_scores.csv")
    under = pd.read_csv(root / "results/tables/under_realizer_diagnostics.csv")
    benchmarks = pd.read_csv(root / "results/tables/benchmark_comparison.csv")
    ablation = pd.read_csv(root / "results/tables/block_ablation_validation.csv")
    expected_diag = pd.read_csv(root / "results/tables/expected_shock_burden_model_diagnostics.csv")

    feature_meta = meta[meta["use_as_feature_in_expanded_score"].astype(bool)]
    blocks = sorted(feature_meta["block"].dropna().unique())
    pipelines = int(potential["n_valid_pipelines"].dropna().max())
    validation_n = int(efficiency["n_validation_countries"].dropna().max())
    ranked_n = int(potential["n_countries_in_sample"].dropna().max())

    assoc = score_audit[
        score_audit["score_variable"].eq("resilience_score_rank_based") & score_audit["method"].eq("spearman")
    ].head(1)
    assoc_line = "Rank-based score association unavailable."
    if not assoc.empty:
        row = assoc.iloc[0]
        assoc_line = (
            f"Rank-based potential score vs 2020-2022 burden: Spearman r={_fmt(row['statistic'])}, "
            f"95% CI {_fmt(row['ci_low'])} to {_fmt(row['ci_high'])}, n={int(row['n'])}."
        )

    win_rows = window[window["method"].eq("spearman")]
    window_line = "; ".join(
        f"{r['outcome_window']} r={_fmt(r['statistic'])} (n={int(r['n'])})" for _, r in win_rows.iterrows()
    )

    profile_counts = profiles.groupby("conversion_profile")["iso3"].nunique().sort_values(ascending=False)
    top_under = efficiency.sort_values("conversion_efficiency_zscore").head(5)
    top_over = efficiency.sort_values("conversion_efficiency_zscore", ascending=False).head(5)
    bottleneck_counts = bottlenecks["weakest_bottleneck_block"].value_counts()
    benchmark_best = benchmarks.sort_values("cross_validated_rmse").head(5)
    ridge = expected_diag[expected_diag["model_version"].eq("ridge")].head(1)
    ridge_line = "Ridge expected-burden model unavailable."
    if not ridge.empty:
        r = ridge.iloc[0]
        ridge_line = f"Primary ridge expected-burden model: CV RMSE={_fmt(r['cv_rmse'], 1)}, adjusted R2={_fmt(r['adjusted_r2'])}, n={int(r['n'])}."

    original = ablation[ablation["model_or_benchmark"].eq("original_minimal_model")]
    all_blocks = ablation[ablation["model_or_benchmark"].eq("all_blocks")]
    if not original.empty and not all_blocks.empty:
        orig_rmse = original.iloc[0]["cross_validated_rmse"]
        all_rmse = all_blocks.iloc[0]["cross_validated_rmse"]
        orig_s = abs(original.iloc[0]["spearman_with_shock_burden"])
        all_s = abs(all_blocks.iloc[0]["spearman_with_shock_burden"])
        if pd.notna(orig_rmse) and pd.notna(all_rmse) and (all_rmse < orig_rmse or all_s > orig_s + 0.05):
            evidence = "A. Expanded blocks improve at least one external-validation or prediction diagnostic."
        else:
            evidence = "B. Static capacity indicators remain weak predictors, but HSR-Convert reveals meaningful conversion gaps and bottleneck profiles."
    else:
        evidence = "B. Static capacity indicators remain weak predictors, but HSR-Convert reveals meaningful conversion gaps and bottleneck profiles."

    lines = [
        "# Run Summary",
        "",
        f"Generated: {now_stamp()}",
        f"Git commit: {_git_hash(root)}",
        "",
        "## Framework",
        "",
        "Stage 2 implements HSR-Convert: Health-System Capacity-to-Resilience Conversion Framework. The main contribution is positive: it measures how pre-shock capacity is converted into shock-realized resilience.",
        "",
        "## Countries, Indicators, and Blocks",
        "",
        f"- Expanded countries in matrix: {len(matrix)}",
        f"- Expanded scoring sample countries: {int(matrix['expanded_primary_sample'].sum())}",
        f"- Validation outcome coverage after scoring: {validation_n} of {ranked_n} countries",
        f"- Usable expanded indicators: {len(feature_meta)}",
        f"- Feature blocks: {', '.join(blocks)}",
        f"- Methodological block-ranking pipelines per block: {pipelines}",
        "",
        "Missingness by block:",
    ]
    for _, row in block_missing.iterrows():
        lines.append(
            f"- {row['block']}: {int(row['n_indicators'])} indicators, median country missing rate {_fmt(row['median_country_missing_rate'])}"
        )
    lines.extend(
        [
            "",
            "## Validation Audit",
            "",
            f"- {assoc_line}",
            f"- Outcome-window sensitivity: {window_line}",
            "",
            "## Potential Capacity and Conversion",
            "",
            f"- {ridge_line}",
            f"- Median conversion efficiency z-score: {_fmt(efficiency['conversion_efficiency_zscore'].median())}",
            f"- Conversion efficiency IQR: {_fmt(efficiency['conversion_efficiency_zscore'].quantile(0.25))} to {_fmt(efficiency['conversion_efficiency_zscore'].quantile(0.75))}",
            "",
            "Conversion profile counts:",
        ]
    )
    for profile, count in profile_counts.items():
        lines.append(f"- {profile}: {count}")
    lines.extend(["", "Top capacity under-realizers / lowest conversion efficiency:"])
    for _, row in top_under.iterrows():
        lines.append(
            f"- {row['country']} ({row['iso3']}): E={_fmt(row['conversion_efficiency_zscore'])}, gap={_fmt(row['realized_resilience_gap'], 1)}"
        )
    lines.extend(["", "Top adaptive over-performers / highest conversion efficiency:"])
    for _, row in top_over.iterrows():
        lines.append(
            f"- {row['country']} ({row['iso3']}): E={_fmt(row['conversion_efficiency_zscore'])}, gap={_fmt(row['realized_resilience_gap'], 1)}"
        )
    lines.extend(["", "## Bottleneck Summary", ""])
    for block, count in bottleneck_counts.items():
        lines.append(f"- {block}: weakest same-income bottleneck for {count} countries")
    lines.extend(["", "Most severe under-realizer diagnostics:"])
    for _, row in under.head(5).iterrows():
        lines.append(
            f"- {row['country']} ({row['iso3']}): weakest block {row['weakest_bottleneck_block']}, gap {_fmt(row['weakest_bottleneck_gap'])}"
        )
    lines.extend(["", "## Benchmarks", ""])
    for _, row in benchmark_best.iterrows():
        lines.append(
            f"- {row['model_or_benchmark']}: CV RMSE={_fmt(row['cross_validated_rmse'], 1)}, Spearman={_fmt(row['spearman_with_shock_burden'])}, AUC={_fmt(row['top_risk_quartile_auc'])}"
        )
    lines.extend(
        [
            "",
            "## Evidence Statement",
            "",
            f"- {evidence}",
            "",
            "## Skipped Datasets and Failed Experiments",
            "",
            "- GHS Index 2019 was not loaded because no legitimate local `data/raw/ghs/ghs_index_2019.csv` file was provided.",
            "- WHO pulse survey, IHME/GBD/HAQ, OECD, and optional clustering were not run in Stage 2.",
            "- Diabetes prevalence was validated through the World Bank API but excluded from scoring because 2015-2019 coverage was too sparse in the API response.",
            "",
            "## Limitations",
            "",
            "- Excess mortality coverage and mortality-registration quality vary by country.",
            "- Expected-burden models are predictive association models, not causal models.",
            "- Conversion profiles are rule-based decision-support diagnostics and should not be read as medical or policy prescriptions.",
            "- Block scores depend on available World Bank proxies and should be revisited if GHS or WHO pulse survey data are legitimately added.",
            "",
            "## Next Steps",
            "",
            "1. Add legitimate GHS 2019 data and rerun preparedness-block scoring and benchmarks.",
            "2. Add WHO pulse survey disruption outcomes as secondary conversion validation.",
            "3. Run Monte Carlo data uncertainty around conversion efficiency and bottleneck stability.",
            "4. Only then consider preliminary clustering as a secondary profile audit.",
        ]
    )
    return write_text("\n".join(lines), root / "RUN_SUMMARY.md")
