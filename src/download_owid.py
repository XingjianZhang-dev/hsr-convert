from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .utils import ROOT, ensure_dirs, now_stamp, offline_mode, save_csv


OWID_EXCESS_MORTALITY_URL = "https://ourworldindata.org/grapher/cumulative-excess-deaths-per-million-covid.csv"
USER_AGENT = "sv-hsrdss-research/0.1 (reproducible academic pipeline)"


def download_owid_excess_mortality(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, object]]:
    ensure_dirs(root)
    raw_path = root / "data/raw/owid/cumulative-excess-deaths-per-million-covid.csv"
    reused_archive = offline_mode() and raw_path.exists()
    if not reused_archive:
        response = requests.get(OWID_EXCESS_MORTALITY_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        response.raise_for_status()
        raw_path.write_text(response.text, encoding="utf-8")

    raw = pd.read_csv(raw_path)
    raw["Day"] = pd.to_datetime(raw["Day"], errors="coerce")
    raw = raw.rename(
        columns={
            "Entity": "country",
            "Code": "iso3",
            "Day": "date",
            "cum_excess_per_million_proj_all_ages": "cumulative_excess_deaths_per_million",
        }
    )
    raw = raw[raw["iso3"].astype(str).str.len() == 3].copy()
    shock = raw[(raw["date"] >= "2020-01-01") & (raw["date"] <= "2022-12-31")].copy()
    shock = shock.dropna(subset=["cumulative_excess_deaths_per_million"])
    latest = (
        shock.sort_values(["iso3", "date"])
        .groupby("iso3", as_index=False)
        .tail(1)
        .rename(columns={"date": "outcome_date"})
    )
    latest = latest[
        [
            "iso3",
            "country",
            "outcome_date",
            "cumulative_excess_deaths_per_million",
        ]
    ].copy()
    latest["outcome_years"] = "2020-2022"
    latest = latest.rename(
        columns={
            "cumulative_excess_deaths_per_million": "cumulative_excess_deaths_per_million_2020_2022"
        }
    )
    save_csv(latest, root / "data/processed/shock_outcomes_2020_2022.csv")

    coverage = pd.DataFrame(
        [
            {
                "outcome": "cumulative_excess_deaths_per_million_2020_2022",
                "source": "Our World in Data",
                "n_countries": latest["iso3"].nunique(),
                "n_raw_rows": len(raw),
                "n_raw_countries": raw["iso3"].nunique(),
                "n_missing_after_2020_2022_filter": int(raw["iso3"].nunique() - latest["iso3"].nunique()),
                "missing_rate_after_2020_2022_filter": float(1.0 - latest["iso3"].nunique() / max(raw["iso3"].nunique(), 1)),
                "first_observation": shock["date"].min().date().isoformat() if not shock.empty else None,
                "last_observation": shock["date"].max().date().isoformat() if not shock.empty else None,
                "countries_with_2022_observation": int((latest["outcome_date"].dt.year == 2022).sum()),
                "missingness_note": "Coverage is limited to countries available in the OWID cumulative excess mortality grapher dataset.",
            }
        ]
    )
    save_csv(coverage, root / "results/tables/outcome_coverage.csv")

    record = {
        "source_name": "Our World in Data cumulative excess deaths per million",
        "access_method": "OWID Grapher CSV",
        "url": OWID_EXCESS_MORTALITY_URL,
        "programmatic": True,
        "download_timestamp": "archived extract reused" if reused_archive else now_stamp(),
        "variables": "country, ISO-3 code, date, cumulative excess deaths per million",
        "years": "2020-2022 used as validation outcomes",
        "raw_rows": len(raw),
        "output_file": "data/processed/shock_outcomes_2020_2022.csv",
        "license_note": "OWID grapher data are generally CC BY unless stated otherwise; verify metadata before publication.",
        "status": "archived extract processed as validation-only outcome" if reused_archive else "downloaded and processed as validation-only outcome",
    }
    return latest, record


if __name__ == "__main__":
    download_owid_excess_mortality()
