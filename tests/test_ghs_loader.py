from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.load_ghs import load_ghs_2019


def test_ghs_loader_skips_missing_file_without_fabrication(tmp_path: Path) -> None:
    (tmp_path / "data/raw/ghs").mkdir(parents=True)
    df, record = load_ghs_2019(tmp_path)
    assert df.empty
    assert record["status"] == "skipped"


def test_ghs_loader_rejects_malformed_file(tmp_path: Path) -> None:
    path = tmp_path / "data/raw/ghs"
    path.mkdir(parents=True)
    pd.DataFrame({"country": ["A"], "score": [1]}).to_csv(path / "ghs_index_2019.csv", index=False)
    with pytest.raises(ValueError):
        load_ghs_2019(tmp_path)

