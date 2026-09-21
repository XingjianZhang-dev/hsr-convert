from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

from .download_worldbank import BASE_URL, USER_AGENT
from .stage4_common import fit_conversion_for_outcome
from .utils import ROOT, ensure_dirs, offline_mode, save_csv


INDICATOR = "SP.DYN.LE00.IN"
YEARS = [2019, 2020, 2021, 2022]
OUTCOME_COLS = [
    "life_expectancy_loss_2019_2020",
    "life_expectancy_loss_2019_2021",
    "life_expectancy_loss_2019_2022",
    "max_life_expectancy_drop_2020_2022",
]


def _download_life_expectancy(root: Path) -> pd.DataFrame:
    url = f"{BASE_URL}/country/all/indicator/{INDICATOR}"
    local = root / "data/raw/worldbank/SP_DYN_LE00_IN_2019_2022_life_expectancy_loss.csv"
    if offline_mode() and local.exists():
        raw = pd.read_csv(local)
        raw["life_expectancy"] = pd.to_numeric(raw["life_expectancy"], errors="coerce")
        return raw[raw["iso3"].astype(str).str.len().eq(3)].copy()
    response = requests.get(
        url,
        params={"format": "json", "per_page": 20000, "date": "2019:2022"},
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if not (isinstance(data, list) and len(data) > 1):
        raise ValueError(f"Unexpected World Bank response for {INDICATOR}")
    rows = []
    for row in data[1] or []:
        rows.append(
            {
                "iso3": row.get("countryiso3code"),
                "country": (row.get("country") or {}).get("value"),
                "year": int(row.get("date")) if row.get("date") else None,
                "life_expectancy": row.get("value"),
                "source_indicator": INDICATOR,
                "source_provenance": f"{url}?format=json&date=2019:2022",
            }
        )
    raw = pd.DataFrame(rows)
    raw["life_expectancy"] = pd.to_numeric(raw["life_expectancy"], errors="coerce")
    raw = raw[raw["iso3"].astype(str).str.len().eq(3)].copy()
    save_csv(raw, root / "data/raw/worldbank/SP_DYN_LE00_IN_2019_2022_life_expectancy_loss.csv")
    return raw


def build_life_expectancy_loss_outcomes(root: Path = ROOT) -> pd.DataFrame:
    ensure_dirs(root)
    raw = _download_life_expectancy(root)
    pivot = raw.pivot_table(index=["iso3", "country"], columns="year", values="life_expectancy", aggfunc="first").reset_index()
    for year in YEARS:
        if year not in pivot.columns:
            pivot[year] = np.nan
    outcomes = pivot[["iso3", "country"]].copy()
    outcomes["life_expectancy_2019"] = pivot[2019]
    for year in [2020, 2021, 2022]:
        outcomes[f"life_expectancy_{year}"] = pivot[year]
        outcomes[f"life_expectancy_loss_2019_{year}"] = pivot[2019] - pivot[year]
    outcomes["max_life_expectancy_drop_2020_2022"] = pivot[2019] - pivot[[2020, 2021, 2022]].min(axis=1)
    outcomes["higher_is_worse"] = True
    outcomes["source_indicator"] = INDICATOR
    outcomes["source_provenance"] = f"{BASE_URL}/country/all/indicator/{INDICATOR}?format=json&date=2019:2022"
    save_csv(outcomes, root / "data/processed/life_expectancy_loss_outcomes.csv")
    return outcomes


def run_life_expectancy_loss_validation(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    outcomes = build_life_expectancy_loss_outcomes(root)
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    baseline = pd.read_csv(root / "results/tables/conversion_efficiency_scores.csv")
    coverage_rows = []
    validation_rows = []
    profile_frames = []
    for col in OUTCOME_COLS:
        merged = potential[["iso3"]].merge(outcomes[["iso3", col]], on="iso3", how="left")
        coverage = int(merged[col].notna().sum())
        coverage_rows.append(
            {
                "outcome": col,
                "source_indicator": INDICATOR,
                "source_provenance": outcomes["source_provenance"].iloc[0],
                "n": potential["iso3"].nunique(),
                "n_scoring_countries": potential["iso3"].nunique(),
                "coverage_count": coverage,
                "missing_count": int(potential["iso3"].nunique() - coverage),
                "missing_rate": float(1.0 - coverage / max(potential["iso3"].nunique(), 1)),
                "higher_is_worse": True,
            }
        )
        conv, diag = fit_conversion_for_outcome(outcomes[["iso3", col]], col, f"life_expectancy::{col}", root)
        if conv.empty:
            validation_rows.append(diag.iloc[0].to_dict())
            continue
        corr = conv[["iso3", "conversion_efficiency_zscore"]].merge(
            baseline[["iso3", "conversion_efficiency_zscore"]].rename(
                columns={"conversion_efficiency_zscore": "owid_conversion_efficiency_zscore"}
            ),
            on="iso3",
            how="inner",
        )
        under_le = set(conv.nsmallest(10, "conversion_efficiency_zscore")["iso3"])
        under_owid = set(baseline.nsmallest(10, "conversion_efficiency_zscore")["iso3"])
        over_le = set(conv.nlargest(10, "conversion_efficiency_zscore")["iso3"])
        over_owid = set(baseline.nlargest(10, "conversion_efficiency_zscore")["iso3"])
        conv["source_indicator"] = INDICATOR
        conv["source_provenance"] = outcomes["source_provenance"].iloc[0]
        conv["higher_is_worse"] = True
        profile_frames.append(conv)
        validation_rows.append(
            {
                "outcome": col,
                "evidence_variant": f"life_expectancy::{col}",
                "status": "fit",
                "n": len(conv),
                "coverage_count": len(conv),
                "missing_count": int(potential["iso3"].nunique() - len(conv)),
                "missing_rate": float(1.0 - len(conv) / max(potential["iso3"].nunique(), 1)),
                "source_indicator": INDICATOR,
                "source_provenance": outcomes["source_provenance"].iloc[0],
                "conversion_efficiency_correlation_with_owid": spearmanr(
                    corr["conversion_efficiency_zscore"], corr["owid_conversion_efficiency_zscore"]
                ).statistic
                if len(corr) >= 4
                else np.nan,
                "under_realizer_top10_overlap_with_owid": len(under_le & under_owid) / 10,
                "over_performer_top10_overlap_with_owid": len(over_le & over_owid) / 10,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    validation_df = pd.DataFrame(validation_rows)
    save_csv(coverage_df, root / "results/tables/stage4_life_expectancy_coverage.csv")
    save_csv(validation_df, root / "results/tables/stage4_life_expectancy_secondary_validation.csv")
    profiles = pd.concat(profile_frames, ignore_index=True) if profile_frames else validation_df.copy()
    save_csv(profiles, root / "results/tables/stage4_life_expectancy_conversion_profiles.csv")
    if profile_frames:
        primary = profiles[profiles["outcome"].eq("max_life_expectancy_drop_2020_2022")].copy()
        if primary.empty:
            primary = profiles[profiles["outcome"].eq(OUTCOME_COLS[0])].copy()
        under = primary.nsmallest(15, "conversion_efficiency_zscore")
        over = primary.nlargest(15, "conversion_efficiency_zscore")
        case = pd.concat([under.assign(case_type="under_realizer"), over.assign(case_type="over_performer")])
        save_csv(case, root / "results/tables/stage4_life_expectancy_under_over_performers.csv")
    else:
        save_csv(validation_df, root / "results/tables/stage4_life_expectancy_under_over_performers.csv")
    _plot_distribution(outcomes, root)
    _plot_conversion_vs_loss(profiles if profile_frames else pd.DataFrame(), root)
    return coverage_df, validation_df


def _plot_distribution(outcomes: pd.DataFrame, root: Path) -> None:
    plt.figure(figsize=(8, 5))
    for col in OUTCOME_COLS:
        vals = outcomes[col].dropna()
        if not vals.empty:
            plt.hist(vals, bins=30, alpha=0.35, label=col)
    plt.xlabel("Life expectancy loss, years")
    plt.ylabel("Countries")
    plt.title("Stage 4 life expectancy loss distributions")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(root / "results/figures/stage4_life_expectancy_loss_distribution.pdf", bbox_inches="tight")
    plt.close()


def _plot_conversion_vs_loss(profiles: pd.DataFrame, root: Path) -> None:
    if profiles.empty:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No life expectancy conversion profiles", ha="center", va="center")
        plt.axis("off")
    else:
        primary = profiles[profiles["outcome"].eq("max_life_expectancy_drop_2020_2022")].copy()
        if primary.empty:
            primary = profiles.copy()
        plt.figure(figsize=(7, 6))
        plt.scatter(primary["observed_outcome"], primary["conversion_efficiency_zscore"], s=28, alpha=0.75)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Life expectancy loss, years")
        plt.ylabel("Conversion efficiency for life-expectancy outcome")
        plt.title("Stage 4 conversion versus life expectancy loss")
    plt.tight_layout()
    plt.savefig(root / "results/figures/stage4_conversion_vs_life_expectancy_loss.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    run_life_expectancy_loss_validation()
