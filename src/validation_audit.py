from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .normalization import normalize_matrix
from .utils import ROOT, ensure_dirs, feature_columns, feature_directions, indicator_metadata, save_csv


OUTCOME = "cumulative_excess_deaths_per_million_2020_2022"
RANK_SCORE = "resilience_score_rank_based"


def _bootstrap_corr(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    method: str,
    n_boot: int = 800,
    seed: int = 20260624,
) -> tuple[float, float, float, float, int]:
    values = df[[x_col, y_col]].dropna().copy()
    n = len(values)
    if n < 4 or values[x_col].nunique() < 2 or values[y_col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan, n
    if method == "spearman":
        stat, pval = spearmanr(values[x_col], values[y_col])
    elif method == "kendall":
        stat, pval = kendalltau(values[x_col], values[y_col])
    else:
        raise ValueError(method)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = values.iloc[rng.integers(0, n, n)]
        if sample[x_col].nunique() < 2 or sample[y_col].nunique() < 2:
            continue
        if method == "spearman":
            boots.append(spearmanr(sample[x_col], sample[y_col]).statistic)
        else:
            boots.append(kendalltau(sample[x_col], sample[y_col]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(stat), float(lo), float(hi), float(pval), n


def _load_validation_base(root: Path) -> pd.DataFrame:
    consensus = pd.read_csv(root / "results/tables/consensus_rank_distribution.csv")
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    matrix_path = root / "data/processed/pre_shock_indicator_matrix_expanded.csv"
    if not matrix_path.exists():
        matrix_path = root / "data/processed/pre_shock_indicator_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    keep = [c for c in ["iso3", "country", "region", "income_group", "population_total"] if c in matrix.columns]
    controls = matrix[keep].drop_duplicates("iso3")
    validation = consensus.merge(outcomes[["iso3", "outcome_date", OUTCOME]], on="iso3", how="inner")
    validation = validation.merge(controls, on="iso3", how="left", suffixes=("", "_matrix"))
    n_countries = float(consensus["n_countries_in_sample"].max())
    validation[RANK_SCORE] = 1.0 - (validation["median_rank"] - 1.0) / max(n_countries - 1.0, 1.0)
    return validation


def run_score_rank_direction_audit(root: Path = ROOT) -> pd.DataFrame:
    ensure_dirs(root)
    validation = _load_validation_base(root)
    variables = [
        ("median_score", "Higher value means higher potential capacity/resilience score."),
        ("median_rank", "Higher value means lower rank position and therefore less resilient by rank."),
        ("mean_rank", "Higher value means lower rank position and therefore less resilient by rank."),
        ("top_quartile_probability", "Higher value means more often in the top resilience quartile."),
        ("bottom_quartile_probability", "Higher value means more often in the bottom resilience quartile."),
        ("rank_instability_index", "Higher value means more method-sensitive ranking, not more or less resilience."),
        (RANK_SCORE, "Higher value means higher rank-based potential capacity/resilience score."),
    ]
    rows = []
    for x_col, direction_note in variables:
        for method in ["spearman", "kendall"]:
            stat, lo, hi, pval, n = _bootstrap_corr(validation, x_col, OUTCOME, method)
            rows.append(
                {
                    "score_variable": x_col,
                    "outcome": OUTCOME,
                    "higher_value_interpretation": direction_note,
                    "method": method,
                    "n": n,
                    "n_ranked_countries": int(pd.read_csv(root / "results/tables/consensus_rank_distribution.csv").shape[0]),
                    "outcome_missing_count_from_ranked_sample": int(
                        pd.read_csv(root / "results/tables/consensus_rank_distribution.csv").shape[0] - validation["iso3"].nunique()
                    ),
                    "outcome_missing_rate_from_ranked_sample": float(
                        1.0
                        - validation["iso3"].nunique()
                        / max(pd.read_csv(root / "results/tables/consensus_rank_distribution.csv").shape[0], 1)
                    ),
                    "statistic": stat,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": pval,
                }
            )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/score_rank_direction_audit.csv")
    return out


def _outcome_windows(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(root / "data/raw/owid/cumulative-excess-deaths-per-million-covid.csv")
    raw["Day"] = pd.to_datetime(raw["Day"], errors="coerce")
    raw = raw.rename(
        columns={
            "Entity": "country",
            "Code": "iso3",
            "Day": "date",
            "cum_excess_per_million_proj_all_ages": "cumulative_excess_deaths_per_million",
        }
    )
    raw = raw[raw["iso3"].astype(str).str.len().eq(3)].copy()
    windows = {
        "2020_end": "2020-12-31",
        "2021_end": "2021-12-31",
        "2022_end": "2022-12-31",
        "2020_2021": "2021-12-31",
        "2020_2022": "2022-12-31",
    }
    frames = []
    coverage_rows = []
    for window, end_date in windows.items():
        sub = raw[(raw["date"] >= "2020-01-01") & (raw["date"] <= pd.Timestamp(end_date))].dropna(
            subset=["cumulative_excess_deaths_per_million"]
        )
        latest = sub.sort_values(["iso3", "date"]).groupby("iso3", as_index=False).tail(1)
        latest = latest[["iso3", "country", "date", "cumulative_excess_deaths_per_million"]].copy()
        latest["outcome_window"] = window
        latest = latest.rename(columns={"date": "outcome_date", "cumulative_excess_deaths_per_million": "outcome_value"})
        frames.append(latest)
        coverage_rows.append(
            {
                "outcome_window": window,
                "window_end": end_date,
                "n_outcome_countries": latest["iso3"].nunique(),
                "n_raw_countries": raw["iso3"].nunique(),
                "missing_count_from_raw_owid_countries": int(raw["iso3"].nunique() - latest["iso3"].nunique()),
                "missing_rate_from_raw_owid_countries": float(1.0 - latest["iso3"].nunique() / max(raw["iso3"].nunique(), 1)),
                "closest_available_rule": "latest observation before or on window end",
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coverage_rows)


def run_outcome_window_sensitivity(root: Path = ROOT) -> pd.DataFrame:
    consensus = pd.read_csv(root / "results/tables/consensus_rank_distribution.csv")
    n_countries = float(consensus["n_countries_in_sample"].max())
    scores = consensus[["iso3", "median_rank"]].copy()
    scores[RANK_SCORE] = 1.0 - (scores["median_rank"] - 1.0) / max(n_countries - 1.0, 1.0)
    windows_long, coverage = _outcome_windows(root)
    rows = []
    for _, cov in coverage.iterrows():
        window = cov["outcome_window"]
        merged = scores.merge(
            windows_long[windows_long["outcome_window"].eq(window)][["iso3", "outcome_value", "outcome_date"]],
            on="iso3",
            how="inner",
        )
        for method in ["spearman", "kendall"]:
            stat, lo, hi, pval, n = _bootstrap_corr(merged, RANK_SCORE, "outcome_value", method)
            rows.append(
                {
                    "outcome_window": window,
                    "window_end": cov["window_end"],
                    "score_variable": RANK_SCORE,
                    "method": method,
                    "n": n,
                    "n_outcome_countries": cov["n_outcome_countries"],
                    "n_scoring_countries": consensus["iso3"].nunique(),
                    "missing_count_from_scoring_sample": int(consensus["iso3"].nunique() - merged["iso3"].nunique()),
                    "missing_rate_from_scoring_sample": float(1.0 - merged["iso3"].nunique() / max(consensus["iso3"].nunique(), 1)),
                    "statistic": stat,
                    "ci_low": lo,
                    "ci_high": hi,
                    "p_value": pval,
                    "closest_available_rule": cov["closest_available_rule"],
                }
            )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/outcome_window_sensitivity.csv")
    plot = out[out["method"].eq("spearman")].copy()
    plt.figure(figsize=(8, 4.5))
    plt.errorbar(
        plot["outcome_window"],
        plot["statistic"],
        yerr=[plot["statistic"] - plot["ci_low"], plot["ci_high"] - plot["statistic"]],
        fmt="o-",
        color="#2b8cbe",
        capsize=4,
    )
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Spearman r with rank-based capacity score")
    plt.xlabel("Outcome window")
    plt.title("Outcome-window sensitivity")
    plt.tight_layout()
    plt.savefig(root / "results/figures/outcome_window_sensitivity.pdf", bbox_inches="tight")
    plt.close()
    return out


def run_validation_sample_bias(root: Path = ROOT) -> pd.DataFrame:
    consensus = pd.read_csv(root / "results/tables/consensus_rank_distribution.csv")
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    matrix_path = root / "data/processed/pre_shock_indicator_matrix_expanded.csv"
    if not matrix_path.exists():
        matrix_path = root / "data/processed/pre_shock_indicator_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    country_cols = [c for c in ["iso3", "country", "region", "income_group", "population_total"] if c in matrix.columns]
    country_meta = matrix[country_cols].drop_duplicates("iso3")
    scored = consensus[["iso3", "country", "region", "income_group"]].drop_duplicates("iso3")
    scored = scored.merge(country_meta, on="iso3", how="left", suffixes=("", "_matrix"))
    for col in ["country", "region", "income_group"]:
        alt = f"{col}_matrix"
        if alt in scored.columns:
            scored[col] = scored[col].fillna(scored[alt])
    outcome_iso = set(outcomes["iso3"])
    score_iso = set(scored["iso3"])
    rows = []
    for _, row in scored[~scored["iso3"].isin(outcome_iso)].iterrows():
        rows.append(
            {
                "section": "scoring_sample_missing_validation_outcome",
                "iso3": row["iso3"],
                "country": row.get("country"),
                "region": row.get("region"),
                "income_group": row.get("income_group"),
                "population_group": _population_group(row.get("population_total", np.nan)),
                "n": 1,
                "coverage_count": 0,
                "missing_count": 1,
                "missing_rate": 1.0,
            }
        )
    outcome_only = outcomes[~outcomes["iso3"].isin(score_iso)].merge(country_meta, on="iso3", how="left", suffixes=("", "_matrix"))
    for _, row in outcome_only.iterrows():
        rows.append(
            {
                "section": "validation_outcome_missing_scoring_sample",
                "iso3": row["iso3"],
                "country": row.get("country"),
                "region": row.get("region"),
                "income_group": row.get("income_group"),
                "population_group": _population_group(row.get("population_total", np.nan)),
                "n": 1,
                "coverage_count": 0,
                "missing_count": 1,
                "missing_rate": 1.0,
            }
        )
    scored["has_validation_outcome"] = scored["iso3"].isin(outcome_iso)
    scored["population_group"] = scored.get("population_total", pd.Series(np.nan, index=scored.index)).apply(_population_group)
    for group_col in ["region", "income_group", "population_group"]:
        if group_col not in scored.columns:
            continue
        summary = scored.groupby(group_col, dropna=False)["has_validation_outcome"].agg(["size", "sum"]).reset_index()
        for _, row in summary.iterrows():
            n = int(row["size"])
            covered = int(row["sum"])
            rows.append(
                {
                    "section": f"coverage_summary_by_{group_col}",
                    "iso3": "",
                    "country": "",
                    group_col: row[group_col],
                    "n": n,
                    "coverage_count": covered,
                    "missing_count": n - covered,
                    "missing_rate": float(1.0 - covered / max(n, 1)),
                }
            )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/validation_sample_bias.csv")

    fig_rows = scored.groupby(["region", "income_group"], dropna=False)["has_validation_outcome"].mean().reset_index()
    pivot = fig_rows.pivot_table(index="region", columns="income_group", values="has_validation_outcome")
    plt.figure(figsize=(10, max(4, 0.45 * len(pivot))))
    plt.imshow(pivot.fillna(0).values, vmin=0, vmax=1, aspect="auto", cmap="viridis")
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=8)
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right", fontsize=8)
    plt.colorbar(label="Validation outcome coverage rate")
    plt.title("Validation coverage by region and income group")
    plt.tight_layout()
    plt.savefig(root / "results/figures/sample_coverage_by_region_income.pdf", bbox_inches="tight")
    plt.close()
    return out


def _population_group(value: object) -> str:
    try:
        pop = float(value)
    except Exception:
        return "population_unavailable"
    if not np.isfinite(pop):
        return "population_unavailable"
    if pop < 1_000_000:
        return "<1m"
    if pop < 10_000_000:
        return "1m_to_10m"
    if pop < 50_000_000:
        return "10m_to_50m"
    return ">=50m"


def _corr_summary(
    df: pd.DataFrame,
    label: str,
    full_validation_n: int,
    x_col: str = RANK_SCORE,
    y_col: str = OUTCOME,
) -> dict[str, object]:
    stat_s, lo_s, hi_s, p_s, n = _bootstrap_corr(df, x_col, y_col, "spearman", n_boot=500)
    stat_k, lo_k, hi_k, p_k, _ = _bootstrap_corr(df, x_col, y_col, "kendall", n_boot=500)
    return {
        "sensitivity_sample": label,
        "n": n,
        "missing_count_from_full_validation_sample": int(full_validation_n - df[["iso3", x_col, y_col]].dropna()["iso3"].nunique()) if "iso3" in df.columns else np.nan,
        "spearman_r": stat_s,
        "spearman_ci_low": lo_s,
        "spearman_ci_high": hi_s,
        "spearman_p_value": p_s,
        "kendall_tau": stat_k,
        "kendall_ci_low": lo_k,
        "kendall_ci_high": hi_k,
        "kendall_p_value": p_k,
    }


def run_outlier_sensitivity(root: Path = ROOT) -> pd.DataFrame:
    validation = _load_validation_base(root)
    rows = []
    full_n = validation["iso3"].nunique()
    rows.append(_corr_summary(validation, "all_validation_countries", full_n))
    if "population_total" in validation.columns:
        rows.append(_corr_summary(validation[validation["population_total"].ge(1_000_000)], "exclude_population_below_1m", full_n))
    else:
        rows.append(
            {
                "sensitivity_sample": "exclude_population_below_1m",
                "n": 0,
                "missing_count_from_full_validation_sample": validation["iso3"].nunique(),
                "spearman_r": np.nan,
                "spearman_ci_low": np.nan,
                "spearman_ci_high": np.nan,
                "spearman_p_value": np.nan,
                "kendall_tau": np.nan,
                "kendall_ci_low": np.nan,
                "kendall_ci_high": np.nan,
                "kendall_p_value": np.nan,
                "status": "skipped_population_unavailable",
            }
        )
    lo, hi = validation[OUTCOME].quantile([0.025, 0.975])
    rows.append(_corr_summary(validation[validation[OUTCOME].between(lo, hi)], "exclude_top_bottom_2_5pct_outcomes", full_n))
    rows.append(_corr_summary(validation[validation["income_group"].eq("High income")], "high_income_subset", full_n))
    rows.append(_corr_summary(validation[~validation["income_group"].eq("High income")], "non_high_income_subset", full_n))
    for region in sorted(validation["region"].dropna().unique()):
        rows.append(_corr_summary(validation[validation["region"].ne(region)], f"leave_region_out::{region}", full_n))
    out = pd.DataFrame(rows)
    if "status" not in out.columns:
        out["status"] = "fit"
    out["status"] = out["status"].fillna("fit")
    save_csv(out, root / "results/tables/outlier_sensitivity.csv")
    return out


def run_indicator_outcome_diagnostic(root: Path = ROOT) -> pd.DataFrame:
    matrix_path = root / "data/processed/pre_shock_indicator_matrix_expanded.csv"
    metadata_path = root / "data/processed/indicator_metadata_expanded.csv"
    if not matrix_path.exists():
        matrix_path = root / "data/processed/pre_shock_indicator_matrix.csv"
        metadata_path = root / "data/processed/indicator_metadata.csv"
    matrix = pd.read_csv(matrix_path)
    meta = pd.read_csv(metadata_path)
    if "use_as_feature_in_expanded_score" in meta.columns:
        feature_meta = meta[meta["use_as_feature_in_expanded_score"].astype(bool)].copy()
    else:
        feature_meta = meta[meta["use_as_feature"].astype(bool)].copy()
    if "block" not in feature_meta.columns:
        feature_meta["block"] = feature_meta.get("conceptual_dimension", "minimal_dimension")
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    work = matrix.merge(outcomes[["iso3", OUTCOME]], on="iso3", how="inner")
    rows = []
    for _, item in feature_meta.iterrows():
        var = item["variable"]
        if var not in work.columns:
            continue
        pair = work[[var, OUTCOME]].dropna()
        stat, lo, hi, pval, n = _bootstrap_corr(pair.rename(columns={var: "x"}), "x", OUTCOME, "spearman", n_boot=500)
        rows.append(
            {
                "diagnostic_type": "indicator_raw",
                "name": var,
                "block": item.get("block", item.get("conceptual_dimension", "")),
                "direction": item.get("direction", ""),
                "higher_feature_interpretation": "higher raw indicator value follows configured direction separately",
                "n": n,
                "n_validation_countries": work["iso3"].nunique(),
                "missing_count_in_validation_sample": int(work[var].isna().sum()),
                "missing_rate_in_validation_sample": float(work[var].isna().mean()),
                "spearman_r": stat,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": pval,
            }
        )
    feature_cols = [v for v in feature_meta["variable"] if v in work.columns]
    directions = feature_meta.set_index("variable")["direction"].to_dict()
    oriented = normalize_matrix(work.set_index("iso3")[feature_cols], directions, "robust_minmax")
    feature_meta = feature_meta.set_index("variable")
    for block, vars_in_block in feature_meta.groupby("block").groups.items():
        cols = [v for v in vars_in_block if v in oriented.columns]
        if not cols:
            continue
        block_score = oriented[cols].mean(axis=1)
        pair = pd.DataFrame({"block_score": block_score}).join(work.set_index("iso3")[[OUTCOME]]).dropna()
        stat, lo, hi, pval, n = _bootstrap_corr(pair, "block_score", OUTCOME, "spearman", n_boot=500)
        rows.append(
            {
                "diagnostic_type": "block_or_dimension_score",
                "name": block,
                "block": block,
                "direction": "oriented",
                "higher_feature_interpretation": "higher oriented block score means stronger potential capacity or lower vulnerability burden",
                "n": n,
                "n_validation_countries": work["iso3"].nunique(),
                "missing_count_in_validation_sample": int(work[cols].isna().all(axis=1).sum()),
                "missing_rate_in_validation_sample": float(work[cols].isna().all(axis=1).mean()),
                "spearman_r": stat,
                "ci_low": lo,
                "ci_high": hi,
                "p_value": pval,
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/indicator_outcome_diagnostic.csv")
    plot = out[out["diagnostic_type"].eq("indicator_raw")].sort_values("spearman_r")
    plt.figure(figsize=(9, max(7, 0.22 * len(plot))))
    plt.barh(plot["name"], plot["spearman_r"], color="#756bb1")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Spearman r with 2020-2022 shock burden")
    plt.title("Indicator-outcome diagnostic")
    plt.tight_layout()
    plt.savefig(root / "results/figures/indicator_outcome_diagnostic.pdf", bbox_inches="tight")
    plt.close()
    return out


def run_validation_audit(root: Path = ROOT) -> None:
    ensure_dirs(root)
    run_score_rank_direction_audit(root)
    run_outcome_window_sensitivity(root)
    run_validation_sample_bias(root)
    run_outlier_sensitivity(root)
    run_indicator_outcome_diagnostic(root)


if __name__ == "__main__":
    run_validation_audit()
