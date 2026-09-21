from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_indicator_matrix import build_indicator_matrix
from src.download_owid import download_owid_excess_mortality
from src.download_worldbank import download_worldbank_inputs
from src.pipeline_ensemble import run_baseline, run_methodological_ensemble
from src.shock_validation import run_shock_validation
from src.utils import audit_result_files, ensure_dirs, now_stamp, write_source_audit, write_text


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "not available"


def _manual_source_records() -> list[dict[str, object]]:
    return [
        {
            "source_name": "WHO health-system resilience indicator package",
            "access_method": "Manual conceptual framework review",
            "url": "manual source not downloaded by minimal pipeline",
            "programmatic": False,
            "download_timestamp": "not downloaded",
            "variables": "none downloaded; informs dimensions only",
            "years": "not applicable",
            "status": "manual review required before manuscript use",
        },
        {
            "source_name": "GHS Index",
            "access_method": "Manual download or official data portal",
            "url": "manual source not downloaded by minimal pipeline",
            "programmatic": False,
            "download_timestamp": "not downloaded",
            "variables": "none downloaded",
            "years": "2019 intended for pre-shock benchmark if licensed/provided",
            "status": "skipped until real data and license notes are provided",
        },
        {
            "source_name": "WHO pulse survey on continuity of essential health services",
            "access_method": "Manual download if accessible",
            "url": "manual source not downloaded by minimal pipeline",
            "programmatic": False,
            "download_timestamp": "not downloaded",
            "variables": "none downloaded",
            "years": "2020-2022 intended as secondary validation outcome",
            "status": "skipped until real data are provided",
        },
        {
            "source_name": "IHME/GBD or HAQ",
            "access_method": "Manual/API access subject to terms",
            "url": "manual source not downloaded by minimal pipeline",
            "programmatic": False,
            "download_timestamp": "not downloaded",
            "variables": "none downloaded",
            "years": "not used in minimal pipeline",
            "status": "skipped",
        },
    ]


def write_run_summary() -> Path:
    if (ROOT / "results/tables/conversion_efficiency_scores.csv").exists():
        from src.stage2_summary import write_hsr_convert_summary

        return write_hsr_convert_summary(ROOT)

    matrix = pd.read_csv(ROOT / "data/processed/pre_shock_indicator_matrix.csv")
    indicator_meta = pd.read_csv(ROOT / "data/processed/indicator_metadata.csv")
    missing = pd.read_csv(ROOT / "results/tables/missingness_by_indicator.csv")
    consensus = pd.read_csv(ROOT / "results/tables/consensus_rank_distribution.csv")
    pipeline_results = pd.read_csv(ROOT / "results/tables/pipeline_grid_results.csv", usecols=["pipeline_id"])
    outcomes = pd.read_csv(ROOT / "results/tables/outcome_coverage.csv")
    correlations = pd.read_csv(ROOT / "results/tables/shock_validation_correlations.csv")

    feature_meta = indicator_meta[indicator_meta["use_as_feature"].astype(bool)]
    primary = matrix[matrix["primary_sample"].astype(bool)]
    score_corr = correlations[
        (correlations["x_variable"] == "rank_based_resilience_score") & (correlations["method"] == "spearman")
    ]
    if score_corr.empty:
        validation_line = "Shock validation correlation was not available."
    else:
        row = score_corr.iloc[0]
        validation_line = (
            "Rank-based pre-shock resilience had Spearman r="
            f"{row['statistic']:.3f} with 2020-2022 cumulative excess deaths per million "
            f"(bootstrap 95% CI {row['ci_low']:.3f} to {row['ci_high']:.3f}; n={int(row['n'])})."
        )

    top_stable = consensus.sort_values(["top_quartile_probability", "median_rank"], ascending=[False, True]).head(5)
    unstable = consensus.sort_values("method_disagreement_index", ascending=False).head(5)
    lines = [
        "# Run Summary",
        "",
        f"Generated: {now_stamp()}",
        f"Git commit: {_git_hash()}",
        "",
        "## Data Sources Actually Downloaded",
        "",
        "- World Bank WDI/HNP indicators for 2015-2019.",
        "- Our World in Data cumulative excess deaths per million for 2020-2022 validation.",
        "",
        "## Sample and Indicators",
        "",
        f"- Primary sample countries: {len(primary)}",
        f"- Extended sample countries: {int(matrix['extended_sample'].sum())}",
        f"- Feature indicators: {len(feature_meta)}",
        f"- Context/control indicators retained outside scoring: {len(indicator_meta) - len(feature_meta)}",
        f"- Median feature missingness rate in primary sample: {primary['feature_missing_rate'].median():.3f}",
        f"- Highest indicator missingness among configured features: {missing[missing['use_as_feature'].astype(bool)]['missing_rate'].max():.3f}",
        "",
        "## Methodological Ensemble",
        "",
        f"- Valid pipelines: {pipeline_results['pipeline_id'].nunique()}",
        f"- Countries ranked per pipeline: {consensus['n_countries_in_sample'].iloc[0]}",
        "",
        "## Main Validation Outcome",
        "",
        f"- Outcome: cumulative excess deaths per million, latest available observation through 2022.",
        f"- Countries with outcome coverage: {int(outcomes['n_countries'].iloc[0])}",
        f"- {validation_line}",
        "",
        "## Top-Level Findings",
        "",
        "- These findings are associations and external validation checks, not causal estimates.",
        "- Countries with high top-quartile probability are robust to many methodological choices.",
        "- Countries with high method disagreement should not be interpreted through exact point ranks.",
        "",
        "Most consistently top-quartile countries in this run:",
    ]
    for _, row in top_stable.iterrows():
        lines.append(
            f"- {row['country']} ({row['iso3']}): top-quartile probability {row['top_quartile_probability']:.2f}, median rank {row['median_rank']:.1f}"
        )
    lines.extend(["", "Most method-sensitive ranks in this run:"])
    for _, row in unstable.iterrows():
        lines.append(
            f"- {row['country']} ({row['iso3']}): disagreement index {row['method_disagreement_index']:.3f}, rank interval {row['rank_p05']:.1f}-{row['rank_p95']:.1f}"
        )
    lines.extend(
        [
            "",
            "## Failed or Skipped Experiments",
            "",
            "- GHS Index, WHO pulse survey, IHME/GBD/HAQ, and OECD data were not downloaded because the minimal pipeline only uses programmatically accessible World Bank and OWID sources.",
            "- Clustering profiles, GDP-adjusted residual resilience, counterfactual improvement paths, Monte Carlo data uncertainty, and ablations remain next-stage modules.",
            "- `statsmodels` is not installed in the current environment, so validation uses scikit-learn/scipy models and bootstrap intervals.",
            "",
            "## Known Limitations",
            "",
            "- Excess mortality coverage is incomplete and may reflect surveillance and mortality-registration differences.",
            "- MCDM rankings are descriptive decision-support outputs and should be interpreted through rank distributions.",
            "- Missing data imputation materially affects some countries, especially in the middle of the rank distribution.",
            "",
            "## Next Steps",
            "",
            "1. Add licensed GHS 2019 data as a pre-shock benchmark and preparedness feature where permitted.",
            "2. Implement Monte Carlo data uncertainty and GDP-adjusted residual resilience.",
            "3. Add externally validated profile clustering and counterfactual improvement scenarios.",
            "4. Add ablation experiments after the expanded pipeline is stable.",
        ]
    )
    return write_text("\n".join(lines), ROOT / "RUN_SUMMARY.md")


def main() -> None:
    ensure_dirs(ROOT)
    worldbank_df, worldbank_records = download_worldbank_inputs(ROOT)
    _ = worldbank_df
    _, owid_record = download_owid_excess_mortality(ROOT)
    records = worldbank_records + [owid_record] + _manual_source_records()
    write_source_audit(records, ROOT / "results/logs/source_audit.md")
    build_indicator_matrix(ROOT)
    run_baseline(ROOT)
    run_methodological_ensemble(ROOT)
    run_shock_validation(ROOT)
    write_run_summary()
    audit_result_files(ROOT)


if __name__ == "__main__":
    main()
