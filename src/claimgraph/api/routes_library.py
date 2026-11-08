"""Library, papers, analysis jobs, and search."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from ..config import get_settings
from ..db import new_session, session_scope
from ..ingest.library import scan_library, save_upload
from ..llm import check_llm
from ..models_db import ClaimRow, Job, Paper
from ..pipeline import run_analysis
from . import jobs
from .schemas import AnalyzeRequest

router = APIRouter()
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def get_session():
    session = new_session()
    try:
        yield session
    finally:
        session.close()


def claim_json(row: ClaimRow) -> dict:
    return {
        "claim_id": row.claim_id,
        "paper_id": row.paper_id,
        "text": row.text,
        "intervention": row.intervention,
        "outcome": row.outcome,
        "direction": row.direction,
        "conditions": row.conditions or {},
        "hedged": row.hedged,
        "section": row.section,
        "page": row.page,
        "quote": row.quote,
        "quote_verified": row.quote_verified,
    }


def paper_summary(paper: Paper, claim_count: int = 0) -> dict:
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "venue": paper.venue,
        "doi": paper.doi,
        "source": paper.source,
        "parse_status": paper.parse_status,
        "parse_error": paper.parse_error,
        "extract_status": paper.extract_status,
        "chunks_done": paper.chunks_done,
        "chunks_total": paper.chunks_total,
        "claims_dropped": paper.claims_dropped,
        "missing": paper.missing,
        "n_pages": paper.n_pages,
        "claims": claim_count,
    }


def scan_job(context: jobs.JobContext) -> dict:
    with session_scope() as session:
        result = scan_library(
            session,
            progress=lambda done, total, name: context.progress("reading", f"Reading papers {done} of {total}", done, total, name),
        )
        return {"found": result.found, "new": result.new, "failed": result.failed, "removed": result.removed}


@router.post("/library/scan")
def library_scan(background: bool = False, session: Session = Depends(get_session)) -> dict:
    """Import new PDFs from the papers folder, in the background if asked (the UI does)."""
    if background:
        return {"job_id": jobs.submit("scan", scan_job)}
    result = scan_library(session)
    return {"found": result.found, "new": result.new, "failed": result.failed, "removed": result.removed, "new_ids": result.new_ids}


@router.post("/library/upload")
def library_upload(file: UploadFile = File(...), session: Session = Depends(get_session)) -> dict:
    name = file.filename or "upload.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be added.")
    if file.file.read(5) != b"%PDF-":
        raise HTTPException(status_code=400, detail="This file is not a valid PDF.")
    file.file.seek(0, 2)
    if file.file.tell() > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="This PDF is larger than 200 MB.")
    file.file.seek(0)
    path = save_upload(file.file, name)
    result = scan_library(session)
    return {"saved_as": path.name, "new": result.new, "failed": result.failed, "new_ids": result.new_ids}


@router.get("/library")
def library(session: Session = Depends(get_session)) -> dict:
    counts = dict(session.execute(select(ClaimRow.paper_id, func.count()).group_by(ClaimRow.paper_id)).all())
    papers = session.scalars(select(Paper).order_by(Paper.added_at.desc(), Paper.title)).all()
    return {
        "papers": [paper_summary(paper, counts.get(paper.id, 0)) for paper in papers],
        "folder": str(get_settings().papers_dir),
        "pending": sum(1 for paper in papers if paper.parse_status == "parsed" and paper.extract_status == "pending" and not paper.missing),
    }


def _paper_or_404(session: Session, paper_id: str) -> Paper:
    paper = session.scalar(
        select(Paper).options(selectinload(Paper.sections), selectinload(Paper.claims)).where(Paper.id == paper_id)
    )
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found.")
    return paper


@router.get("/papers/{paper_id}")
def paper_detail(paper_id: str, session: Session = Depends(get_session)) -> dict:
    paper = _paper_or_404(session, paper_id)
    claims = sorted(paper.claims, key=lambda row: (row.page or 0, row.quote_span_start or 0))
    return {
        **paper_summary(paper, len(paper.claims)),
        "abstract": paper.abstract,
        "has_pdf": bool(paper.file_path and Path(paper.file_path).exists()),
        "sections": [
            {"name": section.name, "heading": section.heading, "page_start": section.page_start, "char_start": section.char_start}
            for section in paper.sections
            if section.heading
        ],
        "claims_list": [claim_json(row) for row in claims],
    }


@router.get("/papers/{paper_id}/pdf")
def paper_pdf(paper_id: str, session: Session = Depends(get_session)) -> FileResponse:
    paper = session.get(Paper, paper_id)
    if paper is None or not paper.file_path or not Path(paper.file_path).exists():
        raise HTTPException(status_code=404, detail="The PDF file is not available.")
    return FileResponse(paper.file_path, media_type="application/pdf", content_disposition_type="inline")


@router.post("/analyze")
def analyze(request: AnalyzeRequest | None = None) -> dict:
    status = check_llm()
    if status.get("fatal"):
        raise HTTPException(status_code=503, detail=status["detail"])
    options = request or AnalyzeRequest()

    def work(context: jobs.JobContext) -> dict:
        return run_analysis(
            limit=options.limit,
            paper_ids=options.paper_ids,
            progress=context.progress,
            should_cancel=context.should_cancel,
        )

    return {"job_id": jobs.submit("analyze", work)}


def job_json(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "progress": job.progress or {},
        "error": job.error,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
    }


@router.get("/jobs/active")
def job_active() -> dict:
    job = jobs.active_job()
    return {"job": job_json(job) if job else None}


@router.get("/jobs/{job_id}")
def job_status(job_id: int, session: Session = Depends(get_session)) -> dict:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job_json(job)


@router.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: int) -> dict:
    return {"cancelling": jobs.cancel(job_id)}


def _like(term: str) -> str:
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


@router.get("/search")
def search(q: str = Query(default="", max_length=300), session: Session = Depends(get_session)) -> dict:
    term = q.strip()
    if len(term) < 2:
        return {"claims": [], "papers": []}
    pattern = _like(term)
    claims = session.scalars(
        select(ClaimRow)
        .where(or_(ClaimRow.text.like(pattern, escape="\\"), ClaimRow.quote.like(pattern, escape="\\")))
        .limit(25)
    ).all()
    papers = session.scalars(
        select(Paper).where(or_(Paper.title.like(pattern, escape="\\"), Paper.authors.like(pattern, escape="\\"))).limit(15)
    ).all()
    return {"claims": [claim_json(row) for row in claims], "papers": [paper_summary(paper) for paper in papers]}
