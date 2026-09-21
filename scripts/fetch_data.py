"""Fetch inputs that are not redistributed in this repository and verify them against the expected SHA-256 hashes.

By default only the Global Health Security Index raw file is fetched, because its redistribution terms are
not stated by the provider. World Bank and Our World in Data extracts are shipped in ``data/raw`` under their
Creative Commons Attribution terms; pass ``--refresh-open-data`` to re-download them from the source APIs
(values may differ from the archived 2026-06-24 snapshot if the providers revise their series).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HASHES = ROOT / "data" / "EXPECTED_HASHES.json"
USER_AGENT = "hsr-convert/1.0 (reproducible academic pipeline)"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)


def verify(path: Path, expected: str) -> bool:
    actual = sha256(path)
    ok = actual == expected
    status = "OK " if ok else "MISMATCH"
    print(f"[{status}] {path.relative_to(ROOT)}\n         expected {expected}\n         actual   {actual}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="only verify files that are already present")
    parser.add_argument("--refresh-open-data", action="store_true", help="also re-download World Bank and OWID inputs")
    args = parser.parse_args()

    expected = json.loads(HASHES.read_text(encoding="utf-8"))
    all_ok = True
    for rel, meta in expected.items():
        target = ROOT / rel
        if not target.exists():
            if args.verify_only:
                print(f"[MISSING] {rel}")
                all_ok = False
                continue
            print(f"Downloading {meta['url']} -> {rel}")
            fetch(meta["url"], target)
        ok = verify(target, meta["sha256"])
        if not ok:
            print("         The provider may have updated the file; the archived analysis used the hash above. "
                  "Results with a different file are a data update, not a reproduction.")
        all_ok &= ok

    if args.refresh_open_data:
        sys.path.insert(0, str(ROOT))
        from src.download_owid import download_owid_excess_mortality
        from src.download_worldbank import download_worldbank_inputs

        download_worldbank_inputs(ROOT)
        download_owid_excess_mortality(ROOT)
        print("Open-data inputs refreshed from the source APIs (values may differ from the archived snapshot).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
