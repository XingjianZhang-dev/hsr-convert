from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from .utils import ROOT, ensure_dirs, indicator_metadata, now_stamp, offline_mode, save_csv, save_json


BASE_URL = "https://api.worldbank.org/v2"
USER_AGENT = "sv-hsrdss-research/0.1 (reproducible academic pipeline)"


def _get_json(url: str, params: dict[str, Any]) -> list[Any]:
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=60)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"Unexpected World Bank API response for {response.url}")
    return data


def _paged_worldbank_rows(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    first = _get_json(url, {**params, "page": 1})
    meta = first[0]
    rows = first[1] or []
    pages = int(meta.get("pages", 1))
    for page in range(2, pages + 1):
        payload = _get_json(url, {**params, "page": page})
        rows.extend(payload[1] or [])
    return rows


def download_country_metadata(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_dirs(root)
    url = f"{BASE_URL}/country"
    local = root / "data/raw/worldbank/country_metadata.csv"
    if offline_mode() and local.exists():
        df = pd.read_csv(local)
        return df, {
            "source_name": "World Bank country metadata",
            "access_method": "Archived World Bank API extract (offline mode)",
            "url": url,
            "programmatic": True,
            "download_timestamp": "archived extract reused",
            "variables": "country name, ISO-3 code, region, income group",
            "years": "not year-specific",
            "raw_rows": len(df),
            "output_file": "data/raw/worldbank/country_metadata.csv",
            "license_note": "World Bank Open Data terms should be reviewed before publication.",
            "status": "loaded_local_archive",
        }
    rows = _paged_worldbank_rows(url, {"format": "json", "per_page": 400})
    records = []
    for row in rows:
        records.append(
            {
                "iso3": row.get("id"),
                "iso2": row.get("iso2Code"),
                "name": row.get("name"),
                "region_id": (row.get("region") or {}).get("id"),
                "region": (row.get("region") or {}).get("value"),
                "income_group_id": (row.get("incomeLevel") or {}).get("id"),
                "income_group": (row.get("incomeLevel") or {}).get("value"),
                "lending_type": (row.get("lendingType") or {}).get("value"),
                "capital_city": row.get("capitalCity"),
                "longitude": row.get("longitude"),
                "latitude": row.get("latitude"),
            }
        )
    df = pd.DataFrame(records)
    df = df[df["iso3"].astype(str).str.len() == 3].copy()
    save_csv(df, root / "data/raw/worldbank/country_metadata.csv")
    return df, {
        "source_name": "World Bank country metadata",
        "access_method": "World Bank API",
        "url": url,
        "programmatic": True,
        "download_timestamp": now_stamp(),
        "variables": "country name, ISO-3 code, region, income group",
        "years": "not year-specific",
        "raw_rows": len(df),
        "output_file": "data/raw/worldbank/country_metadata.csv",
        "license_note": "World Bank Open Data terms should be reviewed before publication.",
        "status": "downloaded",
    }


def download_indicator(indicator_code: str, start_year: int, end_year: int, root: Path = ROOT) -> pd.DataFrame:
    url = f"{BASE_URL}/country/all/indicator/{indicator_code}"
    safe_code = indicator_code.replace(".", "_")
    local = root / f"data/raw/worldbank/{safe_code}_{start_year}_{end_year}.csv"
    if offline_mode() and local.exists():
        return pd.read_csv(local)
    rows = _paged_worldbank_rows(
        url,
        {
            "format": "json",
            "per_page": 20000,
            "date": f"{start_year}:{end_year}",
        },
    )
    records = []
    for row in rows:
        records.append(
            {
                "country": (row.get("country") or {}).get("value"),
                "iso3": row.get("countryiso3code"),
                "indicator_code": indicator_code,
                "indicator_name": (row.get("indicator") or {}).get("value"),
                "year": int(row.get("date")) if row.get("date") else None,
                "value": row.get("value"),
                "unit": row.get("unit"),
                "obs_status": row.get("obs_status"),
                "decimal": row.get("decimal"),
            }
        )
    df = pd.DataFrame(records)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    safe_code = indicator_code.replace(".", "_")
    save_csv(df, root / f"data/raw/worldbank/{safe_code}_{start_year}_{end_year}.csv")
    return df


def download_worldbank_inputs(root: Path = ROOT, config_path: Path | None = None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    ensure_dirs(root)
    config_path = config_path or root / "config/indicators.yml"
    meta = indicator_metadata(config_path)
    cfg = __import__("yaml").safe_load(config_path.read_text(encoding="utf-8"))
    window = cfg["feature_window"]
    start_year = int(window["start_year"])
    end_year = int(window["end_year"])

    records: list[dict[str, Any]] = []
    _, country_record = download_country_metadata(root)
    records.append(country_record)

    frames = []
    for item in tqdm(meta.to_dict("records"), desc="World Bank indicators"):
        code = item["source_code"]
        df = download_indicator(code, start_year, end_year, root)
        df["variable"] = item["variable"]
        frames.append(df)
        records.append(
            {
                "source_name": f"World Bank indicator {code}",
                "access_method": "World Bank Indicators API",
                "url": f"{BASE_URL}/country/all/indicator/{code}?format=json&date={start_year}:{end_year}",
                "programmatic": True,
                "download_timestamp": "archived extract reused" if offline_mode() else now_stamp(),
                "variables": item["variable"],
                "years": f"{start_year}-{end_year}",
                "raw_rows": len(df),
                "output_file": f"data/raw/worldbank/{code.replace('.', '_')}_{start_year}_{end_year}.csv",
                "license_note": "World Bank Open Data terms should be reviewed before publication.",
                "status": "loaded_local_archive" if offline_mode() else "downloaded",
            }
        )
    all_df = pd.concat(frames, ignore_index=True)
    save_csv(all_df, root / f"data/raw/worldbank/worldbank_indicators_{start_year}_{end_year}.csv")
    save_json(records, root / "results/logs/worldbank_download_records.json")
    return all_df, records


if __name__ == "__main__":
    download_worldbank_inputs()
