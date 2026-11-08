"""Does the pipeline recover the planted claims and relations? Run: python -m claimgraph.eval.fixture_eval

Runs in isolation: a temporary folder, and the claimgraph_test database (or SQLite if MySQL is not
configured). The real database, the real vector index and the user's papers are never touched.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from rapidfuzz import fuzz
from sqlalchemy import select

from ..config import ROOT, get_settings, load_settings, reset_settings
from ..db import init_db, reset_engine, session_scope
from ..models_db import Base, ClaimPair, ClaimRow, Paper
from .common import save_result

FIXTURES = ROOT / "tests" / "fixtures"


def match_planted(expected: dict, papers_by_file: dict[str, str], claims: list[dict]) -> dict[str, str | None]:
    """Map each planted claim key to the stored claim that quotes it."""
    found: dict[str, str | None] = {}
    for paper in expected["papers"]:
        paper_id = papers_by_file.get(paper["file"])
        for planted in paper["claims"]:
            best, best_score = None, 0.0
            for claim in claims:
                if claim["paper_id"] != paper_id:
                    continue
                score = fuzz.partial_ratio(planted["quote"].lower(), claim["quote"].lower())
                if score > best_score:
                    best, best_score = claim["claim_id"], score
            found[planted["key"]] = best if best_score >= 90 else None
    return found


def compare_with_expected(expected: dict, papers_by_file: dict[str, str], claims: list[dict], pairs: list[dict]) -> dict:
    planted = match_planted(expected, papers_by_file, claims)
    index = {frozenset((pair["claim_a"], pair["claim_b"])): pair for pair in pairs}
    relations = []
    for item in expected["relations"]:
        a, b = planted.get(item["a"]), planted.get(item["b"])
        pair = index.get(frozenset((a, b))) if a and b else None
        relations.append(
            {
                "a": item["a"],
                "b": item["b"],
                "expected": item["relation"],
                "got": pair["relation"] if pair else None,
                "nli_label": pair["nli_label"] if pair else None,
                "nli_prob": round(pair["nli_prob"], 3) if pair and pair["nli_prob"] is not None else None,
                "conditions_match": pair["conditions_match"] if pair else None,
                "ok": bool(pair and pair["relation"] == item["relation"]),
            }
        )
    return {
        "claims_found": sum(1 for value in planted.values() if value),
        "claims_planted": len(planted),
        "planted": planted,
        "relations": relations,
        "relations_correct": sum(1 for item in relations if item["ok"]),
        "relations_expected": len(relations),
        "all_quotes_verified": all(claim["quote_verified"] for claim in claims),
        "claims_stored": len(claims),
    }


def relations_in_right_section(text: str, relations: list[dict], planted: dict[str, str | None]) -> dict:
    """Does each planted relation appear, with both citations, under the heading for its relation?"""
    from ..report.grounding import HEADINGS

    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("#"):
            current = line.lstrip("# ").strip()
        elif current:
            sections.setdefault(current, []).append(line)
    correct = 0
    for item in relations:
        a, b = planted.get(item["a"]), planted.get(item["b"])
        lines = sections.get(HEADINGS.get(item["expected"], ""), [])
        correct += bool(a and b and any(f"[{a}]" in line and f"[{b}]" in line for line in lines))
    return {"relations_in_right_section": correct, "relations_checked": len(relations)}


def write_reports(outcome: dict) -> dict:
    """Write each topic's report with the real model and check it against the planted relations."""
    from sqlalchemy.orm import selectinload

    from ..llm import unload_llm
    from ..models_db import Topic
    from ..report.report_chain import generate_report

    started = time.perf_counter()
    with session_scope() as session:
        topics = session.scalars(select(Topic).options(selectinload(Topic.claim_links), selectinload(Topic.pairs))).all()
        reports = [generate_report(session, topic) for topic in topics]
        best = {"relations_in_right_section": 0, "relations_checked": len(outcome["relations"])}
        for report in reports:
            found = relations_in_right_section(report.text, outcome["relations"], outcome["planted"])
            if found["relations_in_right_section"] >= best["relations_in_right_section"]:
                best = {**found, "model": report.model, "text": report.text}
        result = {
            **best,
            "reports": len(reports),
            "model_sentences_removed": sum(len(report.removed_sentences or []) for report in reports),
            "seconds": round(time.perf_counter() - started, 1),
        }
    unload_llm()
    return result


def read_outcome() -> tuple[dict[str, str], list[dict], list[dict]]:
    with session_scope() as session:
        papers = {Path(paper.file_path).name: paper.id for paper in session.scalars(select(Paper))}
        claims = [
            {"claim_id": row.claim_id, "paper_id": row.paper_id, "quote": row.quote, "quote_verified": row.quote_verified, "text": row.text}
            for row in session.scalars(select(ClaimRow))
        ]
        pairs = [
            {
                "claim_a": row.claim_a,
                "claim_b": row.claim_b,
                "relation": row.relation,
                "nli_label": row.nli_label,
                "nli_prob": row.nli_prob,
                "conditions_match": row.conditions_match,
            }
            for row in session.scalars(select(ClaimPair))
        ]
    return papers, claims, pairs


def run_fixture_pipeline(progress=None) -> dict:
    """Whole pipeline, real models, isolated storage. Returns the comparison with expected.json."""
    from ..embed.embeddings import release_embeddings
    from ..embed.vectorstore import close_client
    from ..ingest.library import scan_library
    from ..pipeline import run_analysis

    if not (FIXTURES / "expected.json").exists():
        raise FileNotFoundError("Fixtures are missing. Run: python scripts/run.py fixtures")
    expected = json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))

    test_url = load_settings(ROOT).test_mysql_url
    saved = {key: os.environ.get(key) for key in ("CLAIMGRAPH_DATA_DIR", "CLAIMGRAPH_PAPERS_DIR", "DATABASE_URL")}
    workspace = Path(tempfile.mkdtemp(prefix="claimgraph_fixture_"))
    close_client()
    reset_engine()
    try:
        os.environ["CLAIMGRAPH_DATA_DIR"] = str(workspace / "data")
        papers_dir = workspace / "papers"
        papers_dir.mkdir()
        for paper in expected["papers"]:
            shutil.copy(FIXTURES / paper["file"], papers_dir / paper["file"])
        os.environ["CLAIMGRAPH_PAPERS_DIR"] = str(papers_dir)
        os.environ["DATABASE_URL"] = test_url or f"sqlite:///{(workspace / 'fixture.db').as_posix()}"
        reset_settings()
        reset_engine()
        get_settings().ensure_dirs()

        from ..db import get_engine

        engine = get_engine()
        database = "mysql" if engine.url.get_backend_name() == "mysql" else "sqlite"
        Base.metadata.drop_all(engine)  # only ever the test database, chosen above
        init_db()
        with session_scope() as session:
            scan_library(session, papers_dir)

        started = time.perf_counter()
        summary = run_analysis(progress=progress or (lambda *a, **k: None))
        elapsed = time.perf_counter() - started
        papers, claims, pairs = read_outcome()
        outcome = compare_with_expected(expected, papers, claims, pairs)
        outcome.update({"seconds": round(elapsed, 1), "database": database, "summary": summary, "claims": claims})
        outcome["report"] = write_reports(outcome)
        return outcome
    finally:
        release_embeddings()
        close_client()
        reset_engine()
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        reset_settings()
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> Path:
    outcome = run_fixture_pipeline()
    settings = get_settings()
    outcome["llm"] = settings.models["llm"].get("gguf_file") or settings.models["llm"].get("model")
    outcome["status"] = "measured"
    return save_result("fixture_eval", outcome)


if __name__ == "__main__":
    print(main())
