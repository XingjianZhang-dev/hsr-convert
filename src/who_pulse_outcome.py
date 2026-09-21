from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .stage4_common import fit_conversion_for_outcome
from .utils import ROOT, ensure_dirs, save_csv


README = """# WHO Pulse Survey Optional Outcome

Place a legitimate country-level WHO pulse survey file here if available.

Accepted file types: CSV, XLSX, XLS.

The file must contain either ISO3 country codes or country names and at least one numeric disruption measure.
The pipeline will not infer values from PDFs or figures and will not fabricate disruption scores.
"""


def _ensure_folder(root: Path) -> Path:
    folder = root / "data/raw/who_pulse"
    folder.mkdir(parents=True, exist_ok=True)
    readme = folder / "README.md"
    if not readme.exists() or readme.read_text(encoding="utf-8") != README:
        readme.write_text(README, encoding="utf-8")
    return folder


def _placeholder(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _read_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(path)


def _find_iso_col(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        name = str(col).lower()
        if name in {"iso3", "iso_code", "countryiso3code", "code"}:
            return col
    for col in df.columns:
        vals = df[col].dropna().astype(str).str.upper()
        if len(vals) and vals.str.match(r"^[A-Z]{3}$").mean() > 0.8:
            return col
    return None


def run_who_pulse_optional_validation(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    folder = _ensure_folder(root)
    files = [p for p in sorted(folder.glob("*")) if p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    potential_n = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")["iso3"].nunique()
    if not files:
        skipped = pd.DataFrame(
            [
                {
                    "status": "skipped_no_legitimate_file",
                    "file_path": "",
                    "detected_columns": "",
                    "selected_disruption_column": "",
                    "n": 0,
                    "coverage_count": 0,
                    "missing_count": potential_n,
                    "missing_rate": 1.0,
                    "source_provenance": "data/raw/who_pulse/ has no CSV/XLSX/XLS country-level file",
                }
            ]
        )
        save_csv(skipped, root / "results/tables/stage4_who_pulse_inventory.csv")
        save_csv(skipped, root / "results/tables/stage4_who_pulse_secondary_validation.csv")
        save_csv(skipped, root / "results/tables/stage4_who_pulse_profile_overlap.csv")
        _placeholder(root / "results/figures/stage4_who_pulse_profile_validation.pdf", "WHO pulse country-level file not available")
        return skipped, skipped

    inventory_rows = []
    validation_rows = []
    profile_rows = []
    baseline = pd.read_csv(root / "results/tables/conversion_efficiency_scores.csv")
    for path in files:
        df = _read_file(path)
        iso_col = _find_iso_col(df)
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != iso_col]
        selected = numeric_cols[0] if numeric_cols else None
        if iso_col is None or selected is None:
            inventory_rows.append(
                {
                    "status": "skipped_malformed_file",
                    "file_path": str(path.relative_to(root)),
                    "detected_columns": ",".join(map(str, df.columns)),
                    "selected_disruption_column": selected or "",
                    "n": 0,
                    "coverage_count": 0,
                    "missing_count": potential_n,
                    "missing_rate": 1.0,
                    "source_provenance": "file lacks ISO3/code-like column or numeric disruption measure",
                }
            )
            continue
        outcome = df[[iso_col, selected]].rename(columns={iso_col: "iso3", selected: "who_pulse_disruption_score"}).copy()
        outcome["iso3"] = outcome["iso3"].astype(str).str.upper()
        outcome["who_pulse_disruption_score"] = pd.to_numeric(outcome["who_pulse_disruption_score"], errors="coerce")
        conv, diag = fit_conversion_for_outcome(outcome, "who_pulse_disruption_score", "who_pulse", root)
        coverage = int(outcome["who_pulse_disruption_score"].notna().sum())
        inventory_rows.append(
            {
                "status": "loaded" if not conv.empty else "skipped_insufficient_overlap",
                "file_path": str(path.relative_to(root)),
                "detected_columns": ",".join(map(str, df.columns)),
                "selected_disruption_column": str(selected),
                "n": len(df),
                "coverage_count": coverage,
                "missing_count": int(potential_n - coverage),
                "missing_rate": float(1.0 - coverage / max(potential_n, 1)),
                "source_provenance": str(path.relative_to(root)),
            }
        )
        if conv.empty:
            validation_rows.append(diag.iloc[0].to_dict())
            continue
        overlap = conv[["iso3", "conversion_efficiency_zscore"]].merge(
            baseline[["iso3", "conversion_efficiency_zscore"]].rename(columns={"conversion_efficiency_zscore": "owid_efficiency"}),
            on="iso3",
            how="inner",
        )
        validation_rows.append(
            {
                "status": "fit",
                "file_path": str(path.relative_to(root)),
                "outcome": "who_pulse_disruption_score",
                "n": len(conv),
                "coverage_count": len(conv),
                "missing_count": int(potential_n - len(conv)),
                "missing_rate": float(1.0 - len(conv) / max(potential_n, 1)),
                "efficiency_spearman_with_owid": spearmanr(overlap["conversion_efficiency_zscore"], overlap["owid_efficiency"]).statistic
                if len(overlap) >= 4
                else np.nan,
                "source_provenance": str(path.relative_to(root)),
            }
        )
        tmp = conv[["iso3", "country", "conversion_profile", "conversion_efficiency_zscore"]].copy()
        tmp["source_provenance"] = str(path.relative_to(root))
        profile_rows.append(tmp)
    inventory = pd.DataFrame(inventory_rows)
    validation = pd.DataFrame(validation_rows) if validation_rows else inventory.copy()
    save_csv(inventory, root / "results/tables/stage4_who_pulse_inventory.csv")
    save_csv(validation, root / "results/tables/stage4_who_pulse_secondary_validation.csv")
    if profile_rows:
        profiles = pd.concat(profile_rows, ignore_index=True)
        base_profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")[["iso3", "conversion_profile"]].rename(
            columns={"conversion_profile": "baseline_profile"}
        )
        overlap = profiles.merge(base_profiles, on="iso3", how="left")
        overlap["profile_stable_vs_owid"] = overlap["conversion_profile"] == overlap["baseline_profile"]
        overlap["n"] = len(overlap)
        overlap["coverage_count"] = len(overlap)
        overlap["missing_count"] = int(potential_n - len(overlap))
        overlap["missing_rate"] = float(1.0 - len(overlap) / max(potential_n, 1))
        save_csv(overlap, root / "results/tables/stage4_who_pulse_profile_overlap.csv")
        plt.figure(figsize=(7, 4))
        overlap["profile_stable_vs_owid"].value_counts().plot(kind="bar")
        plt.ylabel("Countries")
        plt.title("WHO pulse profile stability vs OWID")
        plt.tight_layout()
        plt.savefig(root / "results/figures/stage4_who_pulse_profile_validation.pdf", bbox_inches="tight")
        plt.close()
    else:
        save_csv(inventory, root / "results/tables/stage4_who_pulse_profile_overlap.csv")
        _placeholder(root / "results/figures/stage4_who_pulse_profile_validation.pdf", "WHO pulse validation skipped")
    return inventory, validation


if __name__ == "__main__":
    run_who_pulse_optional_validation()
