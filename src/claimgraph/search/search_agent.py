"""LangGraph agent for the optional online search: search -> rank -> decide -> expand_references -> END.

Stops at the paper budget or after the configured number of loops. Online papers are stored with
source=online and analysed from their abstract, or from the PDF if arXiv offers one.
"""
from __future__ import annotations

import hashlib
import logging
import re
from typing import TypedDict

import httpx
from langgraph.graph import END, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..ingest.library import _store_parsed, file_sha256, parse_file, paper_id_for
from ..models_db import Paper, Section
from .arxiv_client import search_arxiv
from .openalex import fetch_by_ids, search_works

log = logging.getLogger(__name__)


class SearchState(TypedDict):
    query: str
    records: list[dict]
    frontier: list[str]
    loops: int
    max_loops: int
    budget: int


def _identity(record: dict) -> str:
    if record.get("doi"):
        return "doi:" + record["doi"].lower()
    return "title:" + re.sub(r"[^a-z0-9]+", "", (record.get("title") or "").lower())


def merge_records(existing: list[dict], new: list[dict]) -> list[dict]:
    seen = {_identity(record) for record in existing}
    merged = list(existing)
    for record in new:
        identity = _identity(record)
        if identity not in seen and record.get("title"):
            seen.add(identity)
            merged.append(record)
    return merged


def relevance(record: dict, query: str) -> float:
    """Share of query words in the title and abstract, with a little weight on citations."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    haystack = f"{record.get('title', '')} {record.get('abstract') or ''}".lower()
    overlap = sum(1 for word in words if word in haystack) / (len(words) or 1)
    return overlap + min((record.get("citations") or 0) / 5000, 0.2)


def _search(state: SearchState) -> dict:
    records: list[dict] = []
    for function, name in ((search_works, "OpenAlex"), (search_arxiv, "arXiv")):
        try:
            records.extend(function(state["query"], limit=25))
        except httpx.HTTPError as exc:
            log.warning("%s search failed: %s", name, exc)
    return {"records": merge_records(state["records"], records)}


def _rank(state: SearchState) -> dict:
    ordered = sorted(state["records"], key=lambda record: -relevance(record, state["query"]))
    return {"records": ordered[: state["budget"]]}


def _decide(state: SearchState) -> str:
    if len(state["records"]) >= state["budget"] or state["loops"] >= state["max_loops"]:
        return "end"
    frontier = [ref for record in state["records"] for ref in record.get("references", [])]
    return "expand" if frontier else "end"


def _expand(state: SearchState) -> dict:
    frontier = [ref for record in state["records"][:10] for ref in record.get("references", [])][:50]
    try:
        records = fetch_by_ids(frontier)
    except httpx.HTTPError as exc:
        log.warning("Reference expansion failed: %s", exc)
        records = []
    # Followed references are only kept if they relate to the query at all.
    records = [record for record in records if relevance(record, state["query"]) >= 0.3]
    for record in state["records"]:
        record["references"] = []
    return {"records": merge_records(state["records"], records), "loops": state["loops"] + 1}


def build_agent():
    graph = StateGraph(SearchState)
    graph.add_node("search", _search)
    graph.add_node("rank", _rank)
    graph.add_node("expand_references", _expand)
    graph.set_entry_point("search")
    graph.add_edge("search", "rank")
    graph.add_conditional_edges("rank", _decide, {"expand": "expand_references", "end": END})
    graph.add_edge("expand_references", "rank")
    return graph.compile()


def run_search_agent(query: str) -> list[dict]:
    config = get_settings().pipeline["online_search"]
    state = build_agent().invoke(
        {
            "query": query,
            "records": [],
            "frontier": [],
            "loops": 0,
            "max_loops": int(config.get("max_loops", 3)),
            "budget": int(config.get("max_papers", 50)),
        }
    )
    return state["records"]


def _download_arxiv_pdf(record: dict):
    """Download an arXiv PDF into data/cache/online_pdfs, or return None. Other hosts are not fetched."""
    url = record.get("pdf_url") or ""
    if record.get("origin") != "arxiv" or not url.startswith(("https://arxiv.org/", "http://arxiv.org/")):
        return None
    folder = get_settings().cache_dir / "online_pdfs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{hashlib.sha1(url.encode()).hexdigest()}.pdf"
    if path.exists():
        return path
    try:
        response = httpx.get(url.replace("http://", "https://"), timeout=60, follow_redirects=True)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF-"):
            return None
        path.write_bytes(response.content)
        return path
    except httpx.HTTPError as exc:
        log.warning("Could not download %s: %s", url, exc)
        return None


def add_online_papers(session: Session, records: list[dict], download: bool = True) -> int:
    """Store search results as online papers. Papers the library already holds are skipped."""
    added = 0
    for record in records:
        identity = _identity(record)
        paper_id = "o" + hashlib.sha1(identity.encode()).hexdigest()[:10]
        if session.get(Paper, paper_id) is not None:
            continue
        if record.get("doi") and session.scalar(select(Paper).where(Paper.doi == record["doi"])) is not None:
            continue

        path = _download_arxiv_pdf(record) if download else None
        if path is not None:
            file_hash = file_sha256(path)
            if session.scalar(select(Paper).where(Paper.file_hash == file_hash)) is not None:
                continue
            paper = Paper(id=paper_id_for(file_hash), file_path=str(path), file_hash=file_hash, source="online", title=record["title"][:2000])
            session.add(paper)
            try:
                _store_parsed(session, paper, parse_file(path))
                paper.source = "online"
                added += 1
                session.commit()
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not parse %s: %s", path.name, exc)
                session.rollback()

        abstract = record.get("abstract")
        if not abstract:
            continue
        text = f"Abstract\n{abstract}\n"
        paper = Paper(
            id=paper_id,
            title=record["title"][:2000],
            authors=record.get("authors"),
            year=record.get("year"),
            venue=(record.get("venue") or "")[:512] or None,
            doi=(record.get("doi") or "")[:255] or None,
            abstract=abstract,
            source="online",
            parse_status="parsed",
            n_pages=0,
            page_offsets=[0],
        )
        session.add(paper)
        session.flush()
        session.add(Section(paper_id=paper.id, name="abstract", heading="Abstract", text=text, char_start=0, page_start=1))
        session.commit()
        added += 1
    return added
