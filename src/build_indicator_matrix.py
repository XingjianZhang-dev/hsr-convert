from __future__ import annotations

from pathlib import Path

import pandas as pd

from .plotting import plot_correlation_heatmap, plot_missingness_heatmap
from .utils import ROOT, ensure_dirs, feature_columns, indicator_metadata, load_yaml, save_csv


def _aggregate_indicator(df: pd.DataFrame, variable: str, min_years: int) -> pd.DataFrame:
    sub = df[df["variable"] == variable].dropna(subset=["value"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["iso3", variable, f"{variable}_observed_years", f"{variable}_aggregation"])
    sub["year"] = pd.to_numeric(sub["year"], errors="coerce").astype("Int64")
    rows = []
    for iso3, group in sub.groupby("iso3"):
        group = group.sort_values("year")
        observed = group.dropna(subset=["value"])
        if observed.empty:
            continue
        n_years = observed["year"].nunique()
        if n_years >= min_years:
            value = observed["value"].mean()
            aggregation = f"mean_{int(observed['year'].min())}_{int(observed['year'].max())}"
        else:
            latest = observed.sort_values("year").iloc[-1]
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


def build_indicator_matrix(root: Path = ROOT, config_path: Path | None = None) -> pd.DataFrame:
    ensure_dirs(root)
    config_path = config_path or root / "config/indicators.yml"
    cfg = load_yaml(config_path)
    meta = indicator_metadata(config_path)
    feature_cols = feature_columns(meta)
    min_years = int(cfg["feature_window"]["min_years_for_mean"])
    start_year = int(cfg["feature_window"]["start_year"])
    end_year = int(cfg["feature_window"]["end_year"])

    raw_path = root / f"data/raw/worldbank/worldbank_indicators_{start_year}_{end_year}.csv"
    countries_path = root / "data/raw/worldbank/country_metadata.csv"
    raw = pd.read_csv(raw_path)
    countries = pd.read_csv(countries_path)
    countries = countries[countries["region"].ne("Aggregates")].copy()
    countries = countries[countries["iso3"].astype(str).str.len() == 3].copy()

    matrix = countries[["iso3", "name", "region", "income_group"]].rename(columns={"name": "country"}).drop_duplicates("iso3")
    for variable in meta["variable"]:
        agg = _aggregate_indicator(raw, variable, min_years)
        matrix = matrix.merge(agg, on="iso3", how="left")

    value_cols = meta["variable"].tolist()
    matrix["feature_missing_count"] = matrix[feature_cols].isna().sum(axis=1)
    matrix["feature_missing_rate"] = matrix["feature_missing_count"] / len(feature_cols)
    threshold = float(load_yaml(root / "config/pipeline_grid.yml")["ensemble"]["primary_sample_missingness_threshold"])
    matrix["primary_sample"] = matrix["feature_missing_rate"].le(threshold)
    matrix["extended_sample"] = matrix["feature_missing_rate"].le(0.50)

    coverage_rows = []
    for _, row in meta.iterrows():
        var = row["variable"]
        coverage_rows.append(
            {
                "variable": var,
                "source": row["source"],
                "source_code": row["source_code"],
                "conceptual_dimension": row["conceptual_dimension"],
                "use_as_feature": bool(row["use_as_feature"]),
                "n_countries": int(matrix[var].notna().sum()),
                "n_missing": int(matrix[var].isna().sum()),
                "missing_rate": float(matrix[var].isna().mean()),
                "years_configured": row["years_used"],
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    missing_by_indicator = coverage[["variable", "n_countries", "n_missing", "missing_rate", "use_as_feature"]].copy()
    missing_by_country = matrix[["iso3", "country", "region", "income_group", "feature_missing_count", "feature_missing_rate", "primary_sample", "extended_sample"]].copy()

    save_csv(matrix, root / "data/processed/pre_shock_indicator_matrix.csv")
    save_csv(meta, root / "data/processed/indicator_metadata.csv")
    save_csv(coverage, root / "results/tables/data_coverage.csv")
    save_csv(missing_by_indicator, root / "results/tables/missingness_by_indicator.csv")
    save_csv(missing_by_country, root / "results/tables/missingness_by_country.csv")

    plot_missingness_heatmap(matrix, feature_cols, root / "results/figures/missingness_heatmap.pdf")
    plot_correlation_heatmap(matrix, feature_cols, root / "results/figures/correlation_heatmap.pdf")
    return matrix


if __name__ == "__main__":
    build_indicator_matrix()
