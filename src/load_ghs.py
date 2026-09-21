from __future__ import annotations

from pathlib import Path
import hashlib
import re
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .utils import ROOT, ensure_dirs, save_csv
from .stage4_common import fit_conversion_for_outcome, potential_with_controls


EXPECTED_COLUMNS = [
    "iso3",
    "ghs_overall_2019",
    "ghs_prevention_2019",
    "ghs_detection_2019",
    "ghs_response_2019",
    "ghs_health_system_2019",
    "ghs_compliance_2019",
    "ghs_risk_environment_2019",
]

OFFICIAL_GHS_RAW_URL = "https://ghsindex.org/wp-content/uploads/2022/04/2021-GHS-Index-April-2022.csv"
OFFICIAL_RAW_COLUMNS = {
    "Country": "country_name_raw",
    "Year": "year",
    "OVERALL SCORE": "ghs_overall_2019",
    "1) PREVENTION OF THE EMERGENCE OR RELEASE OF PATHOGENS": "ghs_prevention_2019",
    "2) EARLY DETECTION & REPORTING FOR EPIDEMICS OF POTENTIAL INT'L CONCERN": "ghs_detection_2019",
    "3) RAPID RESPONSE TO AND MITIGATION OF THE SPREAD OF AN EPIDEMIC": "ghs_response_2019",
    "4) SUFFICIENT & ROBUST HEALTH SECTOR TO TREAT THE SICK & PROTECT HEALTH WORKERS": "ghs_health_system_2019",
    "5) COMMITMENTS TO IMPROVING NATIONAL CAPACITY, FINANCING AND ADHERENCE TO NORMS": "ghs_compliance_2019",
    "6) OVERALL RISK ENVIRONMENT AND COUNTRY VULNERABILITY TO BIOLOGICAL THREATS": "ghs_risk_environment_2019",
}
COUNTRY_ALIASES = {
    "bosnia and hercegovina": "bosnia and herzegovina",
    "brunei": "brunei darussalam",
    "congo brazzaville": "congo rep",
    "congo democratic": "congo dem rep",
    "czech republic": "czechia",
    "egypt": "egypt arab rep",
    "iran": "iran islamic rep",
    "laos": "lao pdr",
    "micronesia states of": "micronesia fed sts",
    "north korea": "korea dem peoples rep",
    "russia": "russian federation",
    "slovakia": "slovak republic",
    "somalia": "somalia fed rep",
    "south korea": "korea rep",
    "syria": "syrian arab republic",
    "turkey": "turkiye",
    "united states of america": "united states",
    "venezuela": "venezuela rb",
    "vietnam": "viet nam",
    "yemen": "yemen rep",
}


def load_ghs_2019(root: Path = ROOT) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load optional local GHS 2019 data without fabricating missing values."""

    path = root / "data/raw/ghs/ghs_index_2019.csv"
    if not path.exists():
        return pd.DataFrame(columns=EXPECTED_COLUMNS), {
            "source_name": "GHS Index 2019",
            "status": "skipped",
            "reason": "No local data/raw/ghs/ghs_index_2019.csv file was provided.",
            "programmatic": False,
            "output_file": "not applicable",
        }

    df = pd.read_csv(path, encoding="utf-8-sig")
    if "iso3" in df.columns and "ghs_overall_2019" in df.columns:
        df["iso3"] = df["iso3"].astype(str).str.upper()
        keep_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
        out = df[keep_cols].copy()
        return out, {
            "source_name": "GHS Index 2019",
            "status": "loaded local standardized file",
            "reason": "Local standardized file provided by user.",
            "programmatic": False,
            "output_file": str(path.relative_to(root)),
            "raw_rows": len(out),
        }

    if {"Country", "Year", "OVERALL SCORE"}.issubset(df.columns):
        out = _parse_official_ghs_raw(df, root)
        return out, {
            "source_name": "GHS Index 2019",
            "status": "loaded official raw file",
            "reason": "Official GHS 2019-2021 raw CSV parsed for 2019 rows.",
            "programmatic": False,
            "url": OFFICIAL_GHS_RAW_URL,
            "output_file": str(path.relative_to(root)),
            "raw_rows": len(out),
        }

    if "iso3" not in df.columns or "ghs_overall_2019" not in df.columns:
        raise ValueError("GHS file must include iso3 and ghs_overall_2019 columns.")


def _normalize_country_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", "and")
    text = re.sub(
        r"\b(the|republic of|democratic republic of|people's republic of|plurinational state of|islamic republic of|federated states of)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return COUNTRY_ALIASES.get(text, text)


def _parse_official_ghs_raw(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    missing = [col for col in OFFICIAL_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Official GHS raw file is missing required columns: {missing}")
    ghs = df[df["Year"].eq(2019)].rename(columns=OFFICIAL_RAW_COLUMNS).copy()
    ghs = ghs[list(OFFICIAL_RAW_COLUMNS.values())]
    if ghs.empty:
        raise ValueError("Official GHS raw file contains no 2019 rows.")
    mapping_source = root / "results/tables/potential_capacity_scores.csv"
    if not mapping_source.exists():
        mapping_source = root / "data/processed/pre_shock_indicator_matrix_expanded.csv"
    if not mapping_source.exists():
        raise ValueError("Official GHS raw file requires a project country table for ISO3 matching.")
    countries = pd.read_csv(mapping_source)[["iso3", "country"]].drop_duplicates("iso3")
    country_map = {_normalize_country_name(row.country): row.iso3 for row in countries.itertuples()}
    ghs["iso3"] = ghs["country_name_raw"].map(lambda name: country_map.get(_normalize_country_name(name)))
    score_cols = [col for col in EXPECTED_COLUMNS if col != "iso3"]
    for col in score_cols:
        ghs[col] = pd.to_numeric(ghs[col], errors="coerce")
    return ghs.dropna(subset=["iso3"])[["iso3", *score_cols]].copy()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integrate_ghs_2019(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(root)
    path = root / "data/raw/ghs/ghs_index_2019.csv"
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv")
    if not path.exists():
        audit = pd.DataFrame(
            [
                {
                    "status": "skipped_missing_file",
                    "file_name": str(path.relative_to(root)),
                    "file_hash_sha256": "",
                    "columns_detected": "",
                    "n_countries": 0,
                    "n_matched_iso3_codes": 0,
                    "unmatched_countries": "",
                    "missingness_by_ghs_category": "",
                    "coverage_count": 0,
                    "missing_count": potential["iso3"].nunique(),
                    "missing_rate": 1.0,
                    "note": "No legitimate local GHS 2019 file was provided; values were not fabricated.",
                }
            ]
        )
        save_csv(audit, root / "results/tables/ghs_2019_integration_audit.csv")
        skipped = pd.DataFrame(
            [
                {
                    "status": "skipped_missing_file",
                    "comparison": "GHS 2019 integration",
                    "n": 0,
                    "coverage_count": 0,
                    "missing_count": potential["iso3"].nunique(),
                    "missing_rate": 1.0,
                    "note": "Provide data/raw/ghs/ghs_index_2019.csv with iso3 and ghs_overall_2019 to enable this stage.",
                }
            ]
        )
        for output in [
            "preparedness_block_with_ghs.csv",
            "block_ablation_with_ghs.csv",
            "benchmark_comparison_with_ghs.csv",
        ]:
            save_csv(skipped, root / f"results/tables/{output}")
        _placeholder(root / "results/figures/preparedness_block_before_after_ghs.pdf", "GHS 2019 not available")
        _placeholder(root / "results/figures/ghs_vs_conversion_efficiency.pdf", "GHS 2019 not available")
        return audit, skipped

    ghs, ghs_record = load_ghs_2019(root)
    ghs_source_provenance = (
        f"official GHS raw CSV {ghs_record.get('url')}; local file {path.relative_to(root)}; parsed Year == 2019; verify license before publication"
        if ghs_record.get("url")
        else f"local standardized GHS 2019 file {path.relative_to(root)}; verify license before publication"
    )
    raw = pd.read_csv(path)
    ghs_cols = [c for c in ghs.columns if c != "iso3"]
    merged = potential[["iso3", "country", "preparedness_score"]].merge(ghs, on="iso3", how="left")
    matched = int(merged["ghs_overall_2019"].notna().sum()) if "ghs_overall_2019" in merged.columns else 0
    unmatched = merged.loc[merged[ghs_cols].isna().all(axis=1), "iso3"].tolist() if ghs_cols else merged["iso3"].tolist()
    missingness = {col: float(merged[col].isna().mean()) for col in ghs_cols}
    audit = pd.DataFrame(
        [
            {
                "status": "loaded",
                "file_name": str(path.relative_to(root)),
                "file_hash_sha256": _hash_file(path),
                "columns_detected": ",".join(raw.columns),
                "n_countries": ghs["iso3"].nunique(),
                "n_matched_iso3_codes": matched,
                "unmatched_countries": ",".join(unmatched[:50]),
                "missingness_by_ghs_category": str(missingness),
                "coverage_count": matched,
                "missing_count": int(potential["iso3"].nunique() - matched),
                "missing_rate": float(1.0 - matched / max(potential["iso3"].nunique(), 1)),
                "note": "Local GHS 2019 file was used; verify source license before manuscript use.",
            }
        ]
    )
    save_csv(audit, root / "results/tables/ghs_2019_integration_audit.csv")

    score_cols = [c for c in ghs_cols if c.startswith("ghs_")]
    for col in score_cols:
        merged[f"{col}_scaled"] = _minmax(merged[col])
    scaled_cols = [f"{c}_scaled" for c in score_cols]
    merged["ghs_only_preparedness_score"] = merged[scaled_cols].mean(axis=1) if scaled_cols else np.nan
    merged["preparedness_proxy_plus_ghs_score"] = merged[["preparedness_score", "ghs_only_preparedness_score"]].mean(axis=1)
    merged["n"] = len(merged)
    merged["coverage_count"] = matched
    merged["missing_count"] = int(len(merged) - matched)
    merged["missing_rate"] = float(1.0 - matched / max(len(merged), 1))
    save_csv(merged, root / "results/tables/preparedness_block_with_ghs.csv")

    outcome_df = merged.merge(outcomes[["iso3", "cumulative_excess_deaths_per_million_2020_2022"]], on="iso3", how="inner")
    rows = []
    for name, col in [
        ("World Bank preparedness proxies only", "preparedness_score"),
        ("GHS 2019 only", "ghs_only_preparedness_score"),
        ("World Bank proxies + GHS 2019", "preparedness_proxy_plus_ghs_score"),
    ]:
        pair = outcome_df[[col, "cumulative_excess_deaths_per_million_2020_2022"]].dropna()
        rows.append(
            {
                "comparison": name,
                "score_column": col,
                "n": len(pair),
                "coverage_count": len(pair),
                "missing_count": int(outcome_df["iso3"].nunique() - len(pair)),
                "missing_rate": float(1.0 - len(pair) / max(outcome_df["iso3"].nunique(), 1)),
                "spearman_with_shock_burden": spearmanr(pair[col], pair["cumulative_excess_deaths_per_million_2020_2022"]).statistic
                if len(pair) >= 4
                else np.nan,
            }
        )
    ablation = pd.DataFrame(rows)
    save_csv(ablation, root / "results/tables/block_ablation_with_ghs.csv")

    bench_rows = rows.copy()
    if "ghs_overall_2019" in outcome_df.columns:
        pair = outcome_df[["ghs_overall_2019", "cumulative_excess_deaths_per_million_2020_2022"]].dropna()
        bench_rows.append(
            {
                "comparison": "GHS 2019 overall benchmark",
                "score_column": "ghs_overall_2019",
                "n": len(pair),
                "coverage_count": len(pair),
                "missing_count": int(outcome_df["iso3"].nunique() - len(pair)),
                "missing_rate": float(1.0 - len(pair) / max(outcome_df["iso3"].nunique(), 1)),
                "spearman_with_shock_burden": spearmanr(pair["ghs_overall_2019"], pair["cumulative_excess_deaths_per_million_2020_2022"]).statistic
                if len(pair) >= 4
                else np.nan,
            }
        )
    benchmark = pd.DataFrame(bench_rows)
    save_csv(benchmark, root / "results/tables/benchmark_comparison_with_ghs.csv")
    _plot_ghs_before_after(merged, root)
    _plot_ghs_conversion(merged.merge(profiles[["iso3", "conversion_efficiency_zscore"]], on="iso3", how="left"), root)
    return audit, benchmark


def _minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        return pd.Series(np.nan, index=series.index)
    return (s - lo) / (hi - lo)


def _placeholder(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.text(0.5, 0.5, message, ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _plot_ghs_before_after(df: pd.DataFrame, root: Path) -> None:
    plot = df.dropna(subset=["preparedness_score", "ghs_only_preparedness_score"]).copy()
    plt.figure(figsize=(7, 6))
    plt.scatter(plot["preparedness_score"], plot["ghs_only_preparedness_score"], s=25, alpha=0.75)
    plt.xlabel("World Bank preparedness proxy score")
    plt.ylabel("GHS 2019 preparedness score")
    plt.title("Preparedness block before and after GHS")
    plt.tight_layout()
    plt.savefig(root / "results/figures/preparedness_block_before_after_ghs.pdf", bbox_inches="tight")
    plt.close()


def _plot_ghs_conversion(df: pd.DataFrame, root: Path) -> None:
    plot = df.dropna(subset=["ghs_only_preparedness_score", "conversion_efficiency_zscore"]).copy()
    plt.figure(figsize=(7, 6))
    plt.scatter(plot["ghs_only_preparedness_score"], plot["conversion_efficiency_zscore"], s=25, alpha=0.75)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("GHS 2019 preparedness score")
    plt.ylabel("Conversion efficiency z-score")
    plt.title("GHS preparedness versus conversion efficiency")
    plt.tight_layout()
    plt.savefig(root / "results/figures/ghs_vs_conversion_efficiency.pdf", bbox_inches="tight")
    plt.close()


def integrate_ghs_2019_stage4(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final Stage 4 GHS integration with required stage4-prefixed outputs."""

    ensure_dirs(root)
    path = root / "data/raw/ghs/ghs_index_2019.csv"
    potential = pd.read_csv(root / "results/tables/potential_capacity_scores.csv")
    baseline_profiles = pd.read_csv(root / "results/tables/conversion_profiles.csv")
    baseline_eff = pd.read_csv(root / "results/tables/conversion_efficiency_scores.csv")
    outcomes = pd.read_csv(root / "data/processed/shock_outcomes_2020_2022.csv").rename(
        columns={"cumulative_excess_deaths_per_million_2020_2022": "owid_excess_mortality_2020_2022"}
    )

    if not path.exists():
        audit = pd.DataFrame(
            [
                {
                    "stage": "stage4",
                    "status": "skipped_missing_file",
                    "file_path": str(path.relative_to(root)),
                    "file_hash_sha256": "",
                    "detected_columns": "",
                    "n_rows": 0,
                    "n_matched_iso3_countries": 0,
                    "unmatched_country_names": "",
                    "missingness_by_ghs_category": "",
                    "n": 0,
                    "coverage_count": 0,
                    "missing_count": potential["iso3"].nunique(),
                    "missing_rate": 1.0,
                    "source_provenance": "manual local file not provided; no values fabricated",
                }
            ]
        )
        save_csv(audit, root / "results/tables/stage4_ghs_2019_integration_audit.csv")
        skipped = pd.DataFrame(
            [
                {
                    "stage": "stage4",
                    "status": "skipped_missing_file",
                    "n": 0,
                    "coverage_count": 0,
                    "missing_count": potential["iso3"].nunique(),
                    "missing_rate": 1.0,
                    "source_provenance": "GHS local file absent",
                    "note": "Provide data/raw/ghs/ghs_index_2019.csv to enable GHS-enhanced conversion.",
                }
            ]
        )
        for name in [
            "stage4_preparedness_block_with_ghs.csv",
            "stage4_ghs_benchmark_comparison.csv",
            "stage4_conversion_profiles_with_ghs.csv",
            "stage4_profile_shift_after_ghs.csv",
        ]:
            save_csv(skipped, root / f"results/tables/{name}")
        _placeholder(root / "results/figures/stage4_preparedness_before_after_ghs.pdf", "GHS 2019 not available")
        _placeholder(root / "results/figures/stage4_ghs_vs_conversion_efficiency.pdf", "GHS 2019 not available")
        return audit, skipped

    raw = pd.read_csv(path, encoding="utf-8-sig")
    ghs, ghs_record = load_ghs_2019(root)
    ghs_source_provenance = (
        f"official GHS raw CSV {ghs_record.get('url')}; local file {path.relative_to(root)}; parsed Year == 2019; verify license before publication"
        if ghs_record.get("url")
        else f"local standardized GHS 2019 file {path.relative_to(root)}; verify license before publication"
    )
    ghs_cols = [c for c in ghs.columns if c != "iso3"]
    merged = potential.merge(ghs, on="iso3", how="left")
    matched = int(merged[ghs_cols].notna().any(axis=1).sum()) if ghs_cols else 0
    unmatched = merged.loc[~merged[ghs_cols].notna().any(axis=1), "country"].dropna().tolist() if ghs_cols else merged["country"].tolist()
    missingness = {col: float(merged[col].isna().mean()) for col in ghs_cols}
    audit = pd.DataFrame(
        [
            {
                "stage": "stage4",
                "status": "loaded",
                "file_path": str(path.relative_to(root)),
                "file_hash_sha256": _hash_file(path),
                "detected_columns": ",".join(raw.columns),
                "n_rows": len(raw),
                "n_matched_iso3_countries": matched,
                "unmatched_country_names": ",".join(unmatched[:80]),
                "missingness_by_ghs_category": str(missingness),
                "n": len(raw),
                "coverage_count": matched,
                "missing_count": int(potential["iso3"].nunique() - matched),
                "missing_rate": float(1.0 - matched / max(potential["iso3"].nunique(), 1)),
                "source_provenance": ghs_source_provenance,
            }
        ]
    )
    save_csv(audit, root / "results/tables/stage4_ghs_2019_integration_audit.csv")

    for col in ghs_cols:
        merged[f"{col}_scaled"] = _minmax(merged[col])
    scaled_cols = [f"{c}_scaled" for c in ghs_cols]
    merged["ghs_only_preparedness_score"] = merged[scaled_cols].mean(axis=1)
    merged["preparedness_proxy_plus_ghs_score"] = merged[["preparedness_score", "ghs_only_preparedness_score"]].mean(axis=1)
    merged["n"] = len(merged)
    merged["coverage_count"] = matched
    merged["missing_count"] = int(len(merged) - matched)
    merged["missing_rate"] = float(1.0 - matched / max(len(merged), 1))
    merged["source_provenance"] = ghs_source_provenance
    save_csv(merged, root / "results/tables/stage4_preparedness_block_with_ghs.csv")

    # Build a GHS-enhanced potential-capacity table by replacing the preparedness component.
    potential_override = potential_with_controls(root)
    ghs_scores = merged[["iso3", "preparedness_proxy_plus_ghs_score"]]
    potential_override = potential_override.merge(ghs_scores, on="iso3", how="left")
    potential_override["preparedness_score"] = potential_override["preparedness_proxy_plus_ghs_score"].fillna(
        potential_override["preparedness_score"]
    )
    potential_override["potential_capacity_score"] = potential_override[
        ["capacity_score", "preparedness_score", "vulnerability_score", "equity_access_score"]
    ].mean(axis=1)
    conv, diag = fit_conversion_for_outcome(
        outcomes[["iso3", "owid_excess_mortality_2020_2022"]],
        "owid_excess_mortality_2020_2022",
        "ghs_enhanced_owid",
        root,
        potential_override=potential_override,
    )
    if conv.empty:
        save_csv(diag, root / "results/tables/stage4_conversion_profiles_with_ghs.csv")
    else:
        conv["source_provenance"] = ghs_source_provenance
        save_csv(conv, root / "results/tables/stage4_conversion_profiles_with_ghs.csv")

    benchmark_rows = []
    outcome_join = merged.merge(outcomes[["iso3", "owid_excess_mortality_2020_2022"]], on="iso3", how="inner")
    for label, col in [
        ("World Bank preparedness proxies only", "preparedness_score"),
        ("GHS 2019 only", "ghs_only_preparedness_score"),
        ("World Bank preparedness proxies + GHS 2019", "preparedness_proxy_plus_ghs_score"),
        ("GHS 2019 overall", "ghs_overall_2019"),
    ]:
        if col not in outcome_join.columns:
            continue
        pair = outcome_join[[col, "owid_excess_mortality_2020_2022"]].dropna()
        benchmark_rows.append(
            {
                "comparison": label,
                "score_column": col,
                "n": len(pair),
                "coverage_count": len(pair),
                "missing_count": int(outcome_join["iso3"].nunique() - len(pair)),
                "missing_rate": float(1.0 - len(pair) / max(outcome_join["iso3"].nunique(), 1)),
                "spearman_with_owid_burden": spearmanr(pair[col], pair["owid_excess_mortality_2020_2022"]).statistic
                if len(pair) >= 4
                else np.nan,
                "source_provenance": ghs_source_provenance,
            }
        )
    save_csv(pd.DataFrame(benchmark_rows), root / "results/tables/stage4_ghs_benchmark_comparison.csv")

    if conv.empty:
        shift = diag.copy()
    else:
        shift = conv[["iso3", "country", "conversion_profile", "conversion_efficiency_zscore"]].merge(
            baseline_profiles[["iso3", "conversion_profile"]].rename(columns={"conversion_profile": "baseline_profile"}),
            on="iso3",
            how="left",
        )
        shift = shift.merge(
            baseline_eff[["iso3", "conversion_efficiency_zscore"]].rename(
                columns={"conversion_efficiency_zscore": "baseline_conversion_efficiency"}
            ),
            on="iso3",
            how="left",
        )
        shift["profile_changed_after_ghs"] = shift["conversion_profile"] != shift["baseline_profile"]
        shift["profile_agreement_after_ghs"] = float((~shift["profile_changed_after_ghs"]).mean())
        shift["efficiency_delta_after_ghs"] = shift["conversion_efficiency_zscore"] - shift["baseline_conversion_efficiency"]
        shift["n"] = len(shift)
        shift["coverage_count"] = len(shift)
        shift["missing_count"] = int(potential["iso3"].nunique() - len(shift))
        shift["missing_rate"] = float(1.0 - len(shift) / max(potential["iso3"].nunique(), 1))
        shift["source_provenance"] = ghs_source_provenance
    save_csv(shift, root / "results/tables/stage4_profile_shift_after_ghs.csv")
    _plot_ghs_before_after_stage4(merged, root)
    _plot_ghs_conversion_stage4(merged.merge(baseline_eff[["iso3", "conversion_efficiency_zscore"]], on="iso3", how="left"), root)
    return audit, shift


def _plot_ghs_before_after_stage4(df: pd.DataFrame, root: Path) -> None:
    plot = df.dropna(subset=["preparedness_score", "ghs_only_preparedness_score"])
    plt.figure(figsize=(7, 6))
    plt.scatter(plot["preparedness_score"], plot["ghs_only_preparedness_score"], s=25, alpha=0.75)
    plt.xlabel("World Bank preparedness proxy score")
    plt.ylabel("GHS 2019 preparedness score")
    plt.title("Stage 4 preparedness before and after GHS")
    plt.tight_layout()
    plt.savefig(root / "results/figures/stage4_preparedness_before_after_ghs.pdf", bbox_inches="tight")
    plt.close()


def _plot_ghs_conversion_stage4(df: pd.DataFrame, root: Path) -> None:
    plot = df.dropna(subset=["ghs_only_preparedness_score", "conversion_efficiency_zscore"])
    plt.figure(figsize=(7, 6))
    plt.scatter(plot["ghs_only_preparedness_score"], plot["conversion_efficiency_zscore"], s=25, alpha=0.75)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("GHS 2019 preparedness score")
    plt.ylabel("Baseline conversion efficiency")
    plt.title("Stage 4 GHS versus conversion efficiency")
    plt.tight_layout()
    plt.savefig(root / "results/figures/stage4_ghs_vs_conversion_efficiency.pdf", bbox_inches="tight")
    plt.close()
