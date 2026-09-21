from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def offline_mode() -> bool:
    """True when HSR_OFFLINE is set, so archived raw extracts are reused instead of re-downloaded."""
    return os.environ.get("HSR_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def ensure_dirs(root: Path = ROOT) -> None:
    for rel in [
        "data/raw/worldbank",
        "data/raw/owid",
        "data/interim",
        "data/processed",
        "results/tables",
        "results/figures",
        "results/logs",
        "results/processed_outputs",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_csv(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=index)
    return out


def save_json(obj: Any, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return out


def write_text(text: str, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def indicator_metadata(config_path: Path | None = None) -> pd.DataFrame:
    cfg = load_yaml(config_path or ROOT / "config/indicators.yml")
    rows = []
    for item in cfg["indicators"]:
        row = dict(item)
        row["years_used"] = ",".join(str(y) for y in item.get("years_used", []))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(metadata: pd.DataFrame) -> list[str]:
    return metadata.loc[metadata["use_as_feature"].astype(bool), "variable"].tolist()


def feature_directions(metadata: pd.DataFrame) -> dict[str, str]:
    return (
        metadata.loc[metadata["use_as_feature"].astype(bool), ["variable", "direction"]]
        .set_index("variable")["direction"]
        .to_dict()
    )


def write_source_audit(records: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or ROOT / "results/logs/source_audit.md"
    lines = [
        "# Source Audit",
        "",
        f"Generated: {now_stamp()}",
        "",
        "This audit records data access attempted by the minimal pipeline. Sources marked as manual were not used as generated inputs.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record.get('source_name', 'Unknown source')}",
                "",
                f"- Access method: {record.get('access_method', 'not recorded')}",
                f"- URL: {record.get('url', 'not recorded')}",
                f"- Downloaded programmatically: {record.get('programmatic', False)}",
                f"- Download timestamp: {record.get('download_timestamp', 'not downloaded')}",
                f"- Variables: {record.get('variables', 'not recorded')}",
                f"- Years: {record.get('years', 'not recorded')}",
                f"- Raw rows: {record.get('raw_rows', 'not applicable')}",
                f"- Output file: {record.get('output_file', 'not applicable')}",
                f"- License/usage note: {record.get('license_note', 'review source terms before publication')}",
                f"- Status: {record.get('status', 'not recorded')}",
                "",
            ]
        )
    return write_text("\n".join(lines), path)


def audit_result_files(root: Path = ROOT) -> Path:
    rows: list[dict[str, Any]] = []
    for subdir in [
        "data/processed",
        "results/tables",
        "results/figures",
        "results/logs",
        "results/processed_outputs",
        "paper_assets/tables",
        "paper_assets/figures",
        "paper_assets/notes",
    ]:
        for path in sorted((root / subdir).glob("*")):
            if path.is_file():
                rows.append(
                    {
                        "path": str(path.relative_to(root)),
                        "bytes": path.stat().st_size,
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                    }
                )
    lines = ["# Results Audit", "", f"Generated: {now_stamp()}", ""]
    if not rows:
        lines.append("No generated result files were found.")
    else:
        lines.extend(["| File | Bytes | Modified |", "|---|---:|---|"])
        for row in rows:
            lines.append(f"| `{row['path']}` | {row['bytes']} | {row['modified']} |")
    return write_text("\n".join(lines), root / "RESULTS_AUDIT.md")
