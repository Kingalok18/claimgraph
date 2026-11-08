"""Download evaluation data into data/datasets. SciFact by default; SciFact-Open only when asked.

BioDivergence is never fetched here, because its licence has to be checked by a person first.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claimgraph.config import get_settings  # noqa: E402

SCIFACT_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SCIFACT_OPEN_URL = "https://scifact.s3.amazonaws.com/release/scifact-open/data.tar.gz"


def _fetch(url: str, target: Path) -> None:
    with httpx.stream("GET", url, timeout=120, follow_redirects=True) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_bytes(1024 * 1024):
                handle.write(chunk)


def _safe_extract(archive: Path, folder: Path) -> None:
    """Extract only regular files that stay inside the folder."""
    root = folder.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (folder / member.name).resolve()
            if member.isfile() and target.is_relative_to(root):
                tar.extract(member, folder, filter="data")


def _collect(archive_url: str, wanted: dict[str, str], target: Path) -> None:
    if all((target / name).exists() for name in wanted.values()):
        print(f"Already present: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        archive = folder / "data.tar.gz"
        print(f"Downloading {archive_url}")
        _fetch(archive_url, archive)
        extracted = folder / "extracted"
        extracted.mkdir()
        _safe_extract(archive, extracted)
        for source_name, target_name in wanted.items():
            matches = list(extracted.rglob(source_name))
            if not matches:
                raise FileNotFoundError(f"{source_name} is not in the archive")
            shutil.copy(matches[0], target / target_name)
    print(f"Saved to {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scifact-open", action="store_true", help="Also fetch SciFact-Open (large)")
    args = parser.parse_args()
    datasets = get_settings().datasets_dir
    _collect(
        SCIFACT_URL,
        {"claims_train.jsonl": "claims_train.jsonl", "claims_dev.jsonl": "claims_dev.jsonl", "corpus.jsonl": "corpus.jsonl"},
        datasets / "scifact",
    )
    if args.scifact_open:
        _collect(SCIFACT_OPEN_URL, {"claims.jsonl": "claims.jsonl", "corpus.jsonl": "corpus.jsonl"}, datasets / "scifact_open")


if __name__ == "__main__":
    main()
