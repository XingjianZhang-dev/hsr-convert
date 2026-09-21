from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from .download_worldbank import BASE_URL, USER_AGENT, download_country_metadata
from .imputation import impute_matrix
from .normalization import normalize_matrix
from .utils import ROOT, ensure_dirs, load_yaml, now_stamp, offline_mode, save_csv, save_json
from .weighting import compute_weights
from .load_ghs import load_ghs_2019


def load_block_config(root: Path = ROOT) -> dict[str, Any]:
    return load_yaml(root / "config/indicator_blocks.yml")


def _worldbank_request(code: str, start_year: int, end_year: int, root: Path = ROOT) -> tuple[bool, pd.DataFrame, str]:
    url = f"{BASE_URL}/country/all/indicator/{code}"
    local = root / f"data/raw/worldbank/{code.replace('.', '_')}_{start_year}_{end_year}_expanded.csv"
    if offline_mode() and local.exists():
        df = pd.read_csv(local)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return True, df, "valid_local_archive"
    response = requests.get(
        url,
        params={"format": "json", "per_page": 20000, "date": f"{start_year}:{end_year}"},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if not (isinstance(data, list) and len(data) > 1 and isinstance(data[0], dict) and "pages" in data[0]):
        return False, pd.DataFrame(), str(data)[:300]
    rows = []
    for row in data[1] or []:
        rows.append(
            {
                "country": (row.get("country") or {}).get("value"),
                "iso3": row.get("countryiso3code"),
                "indicator_code": code,
                "indicator_name": (row.get("indicator") or {}).get("value"),
                "year": int(row.get("date")) if row.get("date") else None,
                "value": row.get("value"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return True, df, "valid"


def _aggregate_indicator(df: pd.DataFrame, variable: str, min_years: int) -> pd.DataFrame:
    sub = df.dropna(subset=["value"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["iso3", variable, f"{variable}_observed_years", f"{variable}_aggregation"])
    rows = []
    for iso3, group in sub.groupby("iso3"):
        if not isinstance(iso3, str) or len(iso3) != 3:
            continue
        group = group.sort_values("year")
        n_years = group["year"].nunique()
        if n_years >= min_years:
            value = group["value"].mean()
            aggregation = f"mean_{int(group['year'].min())}_{int(group['year'].max())}"
        else:
            latest = group.iloc[-1]
            value = latest["value"]
            aggregation = f"latest_pre2020_{int(latest['year'])}"
        rows.append(
            {
                "iso3": iso3,
                variable: value,
                f"{variable}_observed_years": int(n_years),
                f"{variable}_aggregation": aggregation,
            }
        )
    return pd.DataFrame(rows)


def _country_metadata(root: Path) -> pd.DataFrame:
    path = root / "data/raw/worldbank/country_metadata.csv"
    if not path.exists():
        download_country_metadata(root)
    countries = pd.read_csv(path)
    countries = countries[countries["region"].ne("Aggregates")].copy()
    countries = countries[countries["iso3"].astype(str).str.len().eq(3)].copy()
    return countries[["iso3", "name", "region", "income_group"]].rename(columns={"name": "country"}).drop_duplicates("iso3")


def build_expanded_indicator_matrix(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    cfg = load_block_config(root)
    start_year = int(cfg["feature_window"]["start_year"])
    end_year = int(cfg["feature_window"]["end_year"])
    min_years = int(cfg["feature_window"]["min_years_for_mean"])
    min_coverage = float(cfg["feature_window"].get("min_country_coverage_rate", 0.45))
    countries = _country_metadata(root)
    matrix = countries.copy()
    total_countries = len(countries)
    metadata_rows = []
    raw_frames = []
    download_records = []

    for item in tqdm(cfg["indicators"], desc="Expanded World Bank indicators"):
        item = dict(item)
        variable = item["variable"]
        source = item.get("source")
        valid = True
        reason = ""
        coverage_count = 0
        if source == "World Bank":
            valid, raw, reason = _worldbank_request(item["source_code"], start_year, end_year, root)
            if valid:
                raw["variable"] = variable
                raw_frames.append(raw)
                safe_code = item["source_code"].replace(".", "_")
                save_csv(raw, root / f"data/raw/worldbank/{safe_code}_{start_year}_{end_year}_expanded.csv")
                agg = _aggregate_indicator(raw, variable, min_years)
                coverage_count = int(agg[variable].notna().sum()) if variable in agg.columns else 0
                matrix = matrix.merge(agg, on="iso3", how="left")
            else:
                matrix[variable] = np.nan
        else:
            matrix[variable] = np.nan
            valid = False
            reason = f"unsupported source in automated expansion: {source}"

        coverage_rate = coverage_count / max(total_countries, 1)
        requested_feature = bool(item.get("use_as_feature", False))
        included = requested_feature and valid and coverage_rate >= min_coverage
        if requested_feature and valid and coverage_rate < min_coverage:
            reason = f"skipped_sparse_coverage_{coverage_rate:.3f}"
        if not requested_feature:
            reason = "context_control_not_scored"
        metadata_rows.append(
            {
                **item,
                "years_used": ",".join(str(y) for y in item.get("years_used", [])),
                "api_validated": bool(valid),
                "api_validation_note": reason if reason else "valid",
                "n_countries": coverage_count,
                "n_total_countries": total_countries,
                "missing_count": int(total_countries - coverage_count),
                "missing_rate": float(1.0 - coverage_rate),
                "use_as_feature_in_expanded_score": bool(included),
            }
        )
        download_records.append(
            {
                "source_name": f"{source} indicator {item.get('source_code')}",
                "variable": variable,
                "url": f"{BASE_URL}/country/all/indicator/{item.get('source_code')}?format=json&date={start_year}:{end_year}",
                "status": "downloaded" if valid else "skipped_invalid",
                "download_timestamp": now_stamp(),
                "n_countries": coverage_count,
                "note": reason,
            }
        )

    # The baseline scoring frame uses World Bank preparedness proxies only. GHS 2019 enters the
    # pipeline through the Stage 4 GHS-enhanced variant, which replaces the preparedness component;
    # merging it here as well would make the baseline depend on whether the GHS file happens to
    # exist at Stage 2 time and would double-count GHS in the GHS-enhanced comparison.
    include_ghs = bool(cfg.get("baseline_blocks", {}).get("include_ghs_2019", False))
    if include_ghs:
        ghs, ghs_record = load_ghs_2019(root)
    else:
        ghs = pd.DataFrame(columns=["iso3"])
        ghs_record = {
            "source_name": "GHS Index 2019",
            "status": "not merged into baseline blocks",
            "reason": "baseline_blocks.include_ghs_2019 is false; GHS 2019 is used only by the Stage 4 GHS-enhanced variant",
            "programmatic": False,
            "output_file": "not applicable",
        }
    if not ghs.empty:
        matrix = matrix.merge(ghs, on="iso3", how="left")
        for col in [c for c in ghs.columns if c != "iso3"]:
            coverage_count = int(matrix[col].notna().sum())
            metadata_rows.append(
                {
                    "variable": col,
                    "label": col.replace("_", " ").title(),
                    "source": "GHS Index",
                    "source_code": col,
                    "block": "preparedness",
                    "conceptual_dimension": "preparedness",
                    "direction": "beneficial",
                    "years_used": "2019",
                    "actionable": False,
                    "counterfactual_allowed": False,
                    "use_as_feature": True,
                    "api_validated": False,
                    "api_validation_note": "loaded_local_manual_file",
                    "n_countries": coverage_count,
                    "n_total_countries": total_countries,
                    "missing_count": int(total_countries - coverage_count),
                    "missing_rate": float(1.0 - coverage_count / max(total_countries, 1)),
                    "use_as_feature_in_expanded_score": coverage_count / max(total_countries, 1) >= min_coverage,
                }
            )
    download_records.append(ghs_record)

    metadata = pd.DataFrame(metadata_rows)
    feature_cols = metadata.loc[metadata["use_as_feature_in_expanded_score"].astype(bool), "variable"].tolist()
    matrix["expanded_feature_missing_count"] = matrix[feature_cols].isna().sum(axis=1)
    matrix["expanded_feature_missing_rate"] = matrix["expanded_feature_missing_count"] / max(len(feature_cols), 1)
    matrix["expanded_primary_sample"] = matrix["expanded_feature_missing_rate"].le(0.35)

    save_csv(matrix, root / "data/processed/pre_shock_indicator_matrix_expanded.csv")
    save_csv(metadata, root / "data/processed/indicator_metadata_expanded.csv")
    save_csv(metadata, root / "results/tables/expanded_indicator_coverage.csv")
    save_json(download_records, root / "results/logs/expanded_worldbank_download_records.json")
    if raw_frames:
        save_csv(pd.concat(raw_frames, ignore_index=True), root / "data/raw/worldbank/worldbank_indicators_expanded_2015_2019.csv")

    _write_block_missingness(matrix, metadata, root)
    _plot_block_missingness(matrix, metadata, root)
    _plot_block_correlation(matrix, metadata, root)
    compute_block_scores(root)
    return matrix, metadata


def _included_features(metadata: pd.DataFrame) -> pd.DataFrame:
    return metadata[metadata["use_as_feature_in_expanded_score"].astype(bool)].copy()


def _write_block_missingness(matrix: pd.DataFrame, metadata: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for block, block_meta in _included_features(metadata).groupby("block"):
        cols = block_meta["variable"].tolist()
        missing_rates = matrix[cols].isna().mean(axis=1)
        rows.append(
            {
                "block": block,
                "n_indicators": len(cols),
                "n_countries": len(matrix),
                "mean_country_missing_rate": float(missing_rates.mean()),
                "median_country_missing_rate": float(missing_rates.median()),
                "countries_with_complete_block": int(missing_rates.eq(0).sum()),
                "countries_missing_entire_block": int(missing_rates.eq(1).sum()),
                "max_country_missing_rate": float(missing_rates.max()),
            }
        )
    out = pd.DataFrame(rows)
    save_csv(out, root / "results/tables/block_missingness.csv")
    return out


def _plot_block_missingness(matrix: pd.DataFrame, metadata: pd.DataFrame, root: Path) -> None:
    feature_meta = _included_features(metadata)
    cols = feature_meta["variable"].tolist()
    data = matrix.set_index("iso3")[cols].isna().astype(int)
    data = data.loc[data.sum(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(12, max(6, min(18, 0.04 * len(data) + 3))))
    plt.imshow(data.values, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.yticks([])
    plt.xticks(range(len(cols)), cols, rotation=70, ha="right", fontsize=7)
    plt.title("Expanded feature-block missingness")
    plt.colorbar(label="Missing")
    plt.tight_layout()
    plt.savefig(root / "results/figures/block_missingness_heatmap.pdf", bbox_inches="tight")
    plt.close()


def _plot_block_correlation(matrix: pd.DataFrame, metadata: pd.DataFrame, root: Path) -> None:
    feature_meta = _included_features(metadata)
    cols = feature_meta["variable"].tolist()
    corr = matrix[cols].corr(method="spearman", min_periods=20)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(len(cols)), cols, rotation=70, ha="right", fontsize=7)
    plt.yticks(range(len(cols)), cols, fontsize=7)
    plt.title("Expanded block feature Spearman correlations")
    plt.colorbar(im, shrink=0.8)
    plt.tight_layout()
    plt.savefig(root / "results/figures/block_correlation_heatmap.pdf", bbox_inches="tight")
    plt.close()


def _block_point_scores(matrix: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    feature_meta = _included_features(metadata)
    sample = matrix[matrix["expanded_primary_sample"].astype(bool)].copy()
    out = sample[[
        c
        for c in ["iso3", "country", "region", "income_group", "population_total", "gdp_per_capita_ppp", "expanded_feature_missing_count", "expanded_feature_missing_rate"]
        if c in sample.columns
    ]].copy()
    block_score_cols = []
    for block, block_meta in feature_meta.groupby("block"):
        cols = block_meta["variable"].tolist()
        directions = block_meta.set_index("variable")["direction"].to_dict()
        raw = sample.set_index("iso3")[cols]
        imputed = impute_matrix(raw, "median")
        normalized = normalize_matrix(imputed, directions, "robust_minmax")
        score_col = f"{block}_score"
        out = out.merge(normalized.mean(axis=1).rename(score_col).reset_index(), on="iso3", how="left")
        out[f"{block}_rank"] = out[score_col].rank(ascending=False, method="min").astype(int)
        out[f"{block}_n_indicators"] = len(cols)
        out[f"{block}_missing_rate"] = sample[cols].isna().mean(axis=1).values
        block_score_cols.append(score_col)
    out["full_potential_capacity_score"] = out[block_score_cols].mean(axis=1)
    out["full_potential_capacity_rank"] = out["full_potential_capacity_score"].rank(ascending=False, method="min").astype(int)
    out["n_countries_in_expanded_sample"] = len(out)
    out["n_expanded_indicators"] = len(feature_meta)
    return out.sort_values("full_potential_capacity_rank")


def compute_block_scores(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(root / "data/processed/pre_shock_indicator_matrix_expanded.csv")
    metadata = pd.read_csv(root / "data/processed/indicator_metadata_expanded.csv")
    block_scores = _block_point_scores(matrix, metadata)
    save_csv(block_scores, root / "results/tables/block_scores.csv")

    feature_meta = _included_features(metadata)
    sample = matrix[matrix["expanded_primary_sample"].astype(bool)].copy()
    lookup_cols = [
        c
        for c in ["country", "region", "income_group", "expanded_feature_missing_count", "expanded_feature_missing_rate"]
        if c in sample.columns
    ]
    lookup = sample.set_index("iso3")[lookup_cols]
    rows = []
    grids = {
        "imputation": ["median", "knn", "iterative"],
        "normalization": ["minmax", "robust_minmax", "vector", "zscore_minmax"],
        "weighting": ["equal", "entropy", "critic", "std", "merec"],
    }
    block_groups: dict[str, list[str]] = {
        block: block_meta["variable"].tolist() for block, block_meta in feature_meta.groupby("block")
    }
    block_groups["full_potential_capacity"] = feature_meta["variable"].tolist()
    directions_all = feature_meta.set_index("variable")["direction"].to_dict()

    for block, cols in block_groups.items():
        raw = sample.set_index("iso3")[cols]
        dirs = {col: directions_all[col] for col in cols}
        ranks = []
        pipeline_scores = []
        pipeline_id = 0
        for imp, norm, weight in product(grids["imputation"], grids["normalization"], grids["weighting"]):
            pipeline_id += 1
            imputed = impute_matrix(raw, imp)
            normalized = normalize_matrix(imputed, dirs, norm)
            weights = compute_weights(normalized, weight)
            score = normalized.mul(weights, axis=1).sum(axis=1)
            rank = score.rank(ascending=False, method="min")
            ranks.append(rank.rename(f"p{pipeline_id:03d}"))
            pipeline_scores.append(score.rename(f"p{pipeline_id:03d}"))
        rank_df = pd.concat(ranks, axis=1)
        score_df = pd.concat(pipeline_scores, axis=1)
        dist = pd.DataFrame(index=rank_df.index)
        dist["block"] = block
        dist["median_rank"] = rank_df.median(axis=1)
        dist["mean_rank"] = rank_df.mean(axis=1)
        dist["rank_std"] = rank_df.std(axis=1, ddof=0)
        dist["rank_p05"] = rank_df.quantile(0.05, axis=1)
        dist["rank_p95"] = rank_df.quantile(0.95, axis=1)
        dist["median_score"] = score_df.median(axis=1)
        n = len(rank_df)
        dist["top_quartile_probability"] = rank_df.le(np.ceil(n * 0.25)).mean(axis=1)
        dist["bottom_quartile_probability"] = rank_df.gt(np.floor(n * 0.75)).mean(axis=1)
        dist["method_disagreement_index"] = (dist["rank_p95"] - dist["rank_p05"]) / max(n - 1, 1)
        dist["n_valid_pipelines"] = rank_df.shape[1]
        dist["n_countries_in_sample"] = n
        dist["n_indicators"] = len(cols)
        dist = dist.join(lookup).reset_index().rename(columns={"index": "iso3"})
        rows.append(dist)
    distributions = pd.concat(rows, ignore_index=True)
    save_csv(distributions, root / "results/tables/block_rank_distributions.csv")
    return block_scores, distributions


if __name__ == "__main__":
    build_expanded_indicator_matrix()
