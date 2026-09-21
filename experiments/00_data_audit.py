from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.download_owid import download_owid_excess_mortality
from src.download_worldbank import download_worldbank_inputs
from src.utils import ensure_dirs, write_source_audit
from experiments.run_minimal_pipeline import _manual_source_records


if __name__ == "__main__":
    ensure_dirs(ROOT)
    _, wb_records = download_worldbank_inputs(ROOT)
    _, owid_record = download_owid_excess_mortality(ROOT)
    write_source_audit(wb_records + [owid_record] + _manual_source_records(), ROOT / "results/logs/source_audit.md")
