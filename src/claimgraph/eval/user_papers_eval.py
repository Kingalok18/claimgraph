"""Analyse a sample of the user's papers on the real database and log what happened (spec M3, M4).

Run: python -m claimgraph.eval.user_papers_eval

The sample shares a subject, so that claims can meet across papers. Safe to run again: finished work
is skipped. The PDFs are only read, and their hashes are compared before and after to prove it.
"""
from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import init_db, session_scope
from ..ingest.library import scan_library
from ..models_db import ClaimPair, ClaimRow, Paper, Topic
from .common import save_result

DEFAULT_SAMPLE = [
    "ADCD-Net",
    "TSFE-Net",
    "Towards_Robust_Tampered_Text",
    "Document_Image_Forgery_Detection",
    "Intelligent_Multimodal",
]


def hash_files(folder: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(folder.rglob("*.pdf")):
        hashes[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def pick_sample(session: Session, names: list[str]) -> list[str]:
    """Paper ids whose file names contain the given text. A name that matches nothing is an error."""
    papers = session.scalars(select(Paper).order_by(Paper.file_path)).all()
    picked: list[str] = []
    for name in names:
        match = next((paper for paper in papers if name in Path(paper.file_path).name), None)
        if match is None:
            raise ValueError(f"No paper in the library has {name!r} in its file name.")
        picked.append(match.id)
    return picked


def summarise(session: Session, paper_ids: list[str]) -> dict:
    """Counts read from the database, so they describe what is stored rather than what a run reported."""
    per_paper = []
    for paper_id in paper_ids:
        paper = session.get(Paper, paper_id)
        kept = session.scalar(select(func.count()).select_from(ClaimRow).where(ClaimRow.paper_id == paper_id)) or 0
        per_paper.append(
            {
                "file": Path(paper.file_path).name,
                "status": paper.extract_status,
                "chunks_done": paper.chunks_done,
                "chunks_total": paper.chunks_total,
                "claims_kept": kept,
                "claims_dropped": paper.claims_dropped or 0,
            }
        )
    kept = sum(item["claims_kept"] for item in per_paper)
    dropped = sum(item["claims_dropped"] for item in per_paper)
    unverified = session.scalar(
        select(func.count()).select_from(ClaimRow).where(ClaimRow.paper_id.in_(paper_ids), ClaimRow.quote_verified.is_(False))
    )
    relations = Counter(session.scalars(select(ClaimPair.relation)).all())
    return {
        "papers_in_sample": len(paper_ids),
        "claims_kept": kept,
        "claims_dropped": dropped,
        "drop_rate": round(dropped / (kept + dropped), 4) if kept + dropped else None,
        "claims_with_unverified_quote": unverified or 0,
        "topics": session.scalar(select(func.count()).select_from(Topic)) or 0,
        "pairs": sum(relations.values()),
        "pairs_supports": relations.get("supports", 0),
        "pairs_contradicts": relations.get("contradicts", 0),
        "pairs_different_setting": relations.get("different_setting", 0),
        "pairs_unclear": relations.get("unclear", 0),
        "per_paper": per_paper,
    }


def main(names: list[str] | None = None) -> Path:
    from ..pipeline import run_analysis

    settings = get_settings()
    init_db()
    before = hash_files(settings.papers_dir)
    started = time.perf_counter()
    with session_scope() as session:
        scan_library(session, settings.papers_dir)
        paper_ids = pick_sample(session, names or DEFAULT_SAMPLE)
    summary = run_analysis(paper_ids=paper_ids)
    seconds = round(time.perf_counter() - started, 1)
    after = hash_files(settings.papers_dir)
    with session_scope() as session:
        counts = summarise(session, paper_ids)
    return save_result(
        "user_papers",
        {
            "status": "measured",
            "sample": "hand-picked to share a subject (document forgery), not a random sample",
            "papers_in_folder": len(before),
            "pdfs_unchanged": before == after,
            "seconds_this_run": seconds,
            "extraction_this_run": summary.get("extraction"),
            **counts,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", action="append", help="Text from a file name. Repeat for each paper.")
    print(main(parser.parse_args().match))
