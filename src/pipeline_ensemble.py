from __future__ import annotations

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from .imputation import impute_matrix
from .mcdm_methods import rank_scores
from .normalization import normalize_matrix
from .plotting import (
    plot_baseline_rankings,
    plot_method_disagreement,
    plot_rank_intervals,
    plot_top_quartile_probability,
)
from .utils import ROOT, ensure_dirs, feature_columns, feature_directions, indicator_metadata, load_yaml, save_csv
from .weighting import compute_weights


def _load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], dict]:
    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix.csv")
    meta = indicator_metadata(root / "config/indicators.yml")
    feature_cols = feature_columns(meta)
    directions = feature_directions(meta)
    grid = load_yaml(root / "config/pipeline_grid.yml")
    return matrix, meta, feature_cols, directions, grid


def _sample_matrix(matrix: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    sample = matrix[matrix["primary_sample"].astype(bool)].copy()
    sample = sample.dropna(how="all", subset=feature_cols)
    return sample


def run_baseline(root: Path = ROOT) -> pd.DataFrame:
    ensure_dirs(root)
    matrix, _meta, feature_cols, directions, grid = _load_inputs(root)
    sample = _sample_matrix(matrix, feature_cols)
    baseline_cfg = grid["baseline"]
    seed = int(grid["ensemble"].get("random_seed", 20260624))

    raw_features = sample.set_index("iso3")[feature_cols]
    imputed = impute_matrix(raw_features, baseline_cfg["imputation"], random_state=seed)
    normalized = normalize_matrix(imputed, directions, baseline_cfg["normalization"])
    weights = compute_weights(normalized, baseline_cfg["weighting"])

    save_csv(imputed.reset_index(), root / f"results/processed_outputs/baseline_imputed_{baseline_cfg['imputation']}.csv")
    save_csv(normalized.reset_index(), root / f"results/processed_outputs/baseline_normalized_{baseline_cfg['normalization']}.csv")
    weight_table = weights.rename("weight").reset_index().rename(columns={"index": "variable"})
    weight_table["n_countries_in_sample"] = len(sample)
    weight_table["n_indicators"] = len(feature_cols)
    weight_table["mean_feature_missing_rate_before_imputation"] = sample["feature_missing_rate"].mean()
    weight_table["max_feature_missing_rate_before_imputation"] = sample["feature_missing_rate"].max()
    save_csv(weight_table, root / "results/tables/baseline_weights.csv")

    rows = []
    lookup = sample.set_index("iso3")[["country", "region", "income_group", "feature_missing_count", "feature_missing_rate"]]
    for ranking_method in baseline_cfg["ranking_methods"]:
        ranked = rank_scores(normalized, weights, ranking_method)
        ranked = ranked.join(lookup)
        ranked = ranked.reset_index().rename(columns={"index": "iso3"})
        ranked["ranking_method"] = ranking_method
        ranked["weighting_method"] = baseline_cfg["weighting"]
        ranked["normalization_method"] = baseline_cfg["normalization"]
        ranked["imputation_method"] = baseline_cfg["imputation"]
        ranked["n_countries_in_sample"] = len(sample)
        ranked["n_indicators"] = len(feature_cols)
        rows.append(ranked)
    baseline = pd.concat(rows, ignore_index=True)
    baseline = baseline[
        [
            "iso3",
            "country",
            "region",
            "income_group",
            "score",
            "rank",
            "ranking_method",
            "weighting_method",
            "normalization_method",
            "imputation_method",
            "feature_missing_count",
            "feature_missing_rate",
            "n_countries_in_sample",
            "n_indicators",
        ]
    ].sort_values(["ranking_method", "rank", "country"])
    save_csv(baseline, root / "results/tables/baseline_rankings.csv")
    plot_baseline_rankings(baseline, root / "results/figures/baseline_top_bottom_ranks.pdf")
    return baseline


def _pairwise_rank_correlations(pipeline_results: pd.DataFrame, root: Path) -> pd.DataFrame:
    ranks = pipeline_results.pivot_table(index="iso3", columns="pipeline_id", values="rank")
    spearman = ranks.corr(method="spearman")
    rows = []
    pipeline_ids = list(ranks.columns)
    missing_by_country = (
        pipeline_results[["iso3", "feature_missing_rate"]]
        .drop_duplicates("iso3")
        .set_index("iso3")["feature_missing_rate"]
    )
    for p1, p2 in combinations(pipeline_ids, 2):
        s = spearman.loc[p1, p2]
        pair = ranks[[p1, p2]].dropna()
        k = kendalltau(pair[p1], pair[p2], nan_policy="omit").statistic
        rows.append(
            {
                "pipeline_a": p1,
                "pipeline_b": p2,
                "spearman_r": s,
                "kendall_tau": k,
                "n_countries_pairwise": len(pair),
                "mean_feature_missing_rate_before_imputation": float(missing_by_country.loc[pair.index].mean()),
                "max_feature_missing_rate_before_imputation": float(missing_by_country.loc[pair.index].max()),
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/pipeline_rank_correlations.csv")
    return out


def run_methodological_ensemble(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    matrix, _meta, feature_cols, directions, grid = _load_inputs(root)
    sample = _sample_matrix(matrix, feature_cols)
    ensemble_cfg = grid["ensemble"]
    seed = int(ensemble_cfg.get("random_seed", 20260624))
    raw_features = sample.set_index("iso3")[feature_cols]
    lookup = sample.set_index("iso3")[["country", "region", "income_group", "feature_missing_count", "feature_missing_rate"]]

    imputed_cache: dict[str, pd.DataFrame] = {}
    normalized_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for imp in ensemble_cfg["imputation_methods"]:
        imputed = impute_matrix(raw_features, imp, random_state=seed)
        imputed_cache[imp] = imputed
        save_csv(imputed.reset_index(), root / f"results/processed_outputs/imputed_{imp}.csv")
        for norm in ensemble_cfg["normalization_methods"]:
            normalized = normalize_matrix(imputed, directions, norm)
            normalized_cache[(imp, norm)] = normalized
            save_csv(normalized.reset_index(), root / f"results/processed_outputs/normalized_{imp}_{norm}.csv")

    rows = []
    pipeline_counter = 0
    for imp, norm, weight_method, rank_method in product(
        ensemble_cfg["imputation_methods"],
        ensemble_cfg["normalization_methods"],
        ensemble_cfg["weighting_methods"],
        ensemble_cfg["ranking_methods"],
    ):
        pipeline_counter += 1
        pipeline_id = f"p{pipeline_counter:04d}_{imp}_{norm}_{weight_method}_{rank_method}"
        normalized = normalized_cache[(imp, norm)]
        weights = compute_weights(normalized, weight_method)
        ranked = rank_scores(normalized, weights, rank_method)
        ranked = ranked.join(lookup).reset_index().rename(columns={"index": "iso3"})
        ranked["pipeline_id"] = pipeline_id
        ranked["imputation_method"] = imp
        ranked["normalization_method"] = norm
        ranked["weighting_method"] = weight_method
        ranked["ranking_method"] = rank_method
        rows.append(ranked)

    pipeline_results = pd.concat(rows, ignore_index=True)
    pipeline_results["n_countries_in_sample"] = len(sample)
    pipeline_results["n_indicators"] = len(feature_cols)
    pipeline_results = pipeline_results[
        [
            "pipeline_id",
            "iso3",
            "country",
            "region",
            "income_group",
            "score",
            "rank",
            "imputation_method",
            "normalization_method",
            "weighting_method",
            "ranking_method",
            "feature_missing_count",
            "feature_missing_rate",
            "n_countries_in_sample",
            "n_indicators",
        ]
    ]
    save_csv(pipeline_results, root / "results/tables/pipeline_grid_results.csv")

    ranks = pipeline_results.pivot_table(index="iso3", columns="pipeline_id", values="rank")
    scores = pipeline_results.pivot_table(index="iso3", columns="pipeline_id", values="score")
    n_countries = len(ranks)
    n_pipelines = ranks.shape[1]
    top10_cut = min(10, n_countries)
    top_quartile_cut = int(np.ceil(n_countries * 0.25))
    bottom_quartile_cut = int(np.floor(n_countries * 0.75))

    consensus = pd.DataFrame(index=ranks.index)
    consensus["median_rank"] = ranks.median(axis=1)
    consensus["mean_rank"] = ranks.mean(axis=1)
    consensus["rank_std"] = ranks.std(axis=1, ddof=0)
    consensus["rank_p05"] = ranks.quantile(0.05, axis=1)
    consensus["rank_p95"] = ranks.quantile(0.95, axis=1)
    consensus["median_score"] = scores.median(axis=1)
    consensus["top10_probability"] = ranks.le(top10_cut).mean(axis=1)
    consensus["top_quartile_probability"] = ranks.le(top_quartile_cut).mean(axis=1)
    consensus["bottom_quartile_probability"] = ranks.gt(bottom_quartile_cut).mean(axis=1)
    consensus["rank_instability_index"] = consensus["rank_std"] / max(n_countries - 1, 1)
    consensus["method_disagreement_index"] = (consensus["rank_p95"] - consensus["rank_p05"]) / max(n_countries - 1, 1)
    consensus["n_valid_pipelines"] = n_pipelines
    consensus = consensus.join(lookup)
    consensus = consensus.reset_index().rename(columns={"index": "iso3"})
    consensus["n_countries_in_sample"] = n_countries
    consensus["n_indicators"] = len(feature_cols)
    consensus = consensus[
        [
            "iso3",
            "country",
            "region",
            "income_group",
            "median_rank",
            "mean_rank",
            "rank_std",
            "rank_p05",
            "rank_p95",
            "median_score",
            "top10_probability",
            "top_quartile_probability",
            "bottom_quartile_probability",
            "rank_instability_index",
            "method_disagreement_index",
            "feature_missing_count",
            "feature_missing_rate",
            "n_valid_pipelines",
            "n_countries_in_sample",
            "n_indicators",
        ]
    ].sort_values(["median_rank", "rank_std", "country"])
    save_csv(consensus, root / "results/tables/consensus_rank_distribution.csv")

    _pairwise_rank_correlations(pipeline_results, root)
    plot_rank_intervals(consensus, root / "results/figures/rank_interval_plot.pdf")
    plot_top_quartile_probability(consensus, root / "results/figures/top_quartile_probability_heatmap.pdf")
    plot_method_disagreement(consensus, root / "results/figures/method_disagreement_plot.pdf")
    return pipeline_results, consensus


if __name__ == "__main__":
    run_baseline()
    run_methodological_ensemble()
