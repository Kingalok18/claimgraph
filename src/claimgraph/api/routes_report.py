"""The evidence-map report, its citations, and its exports."""
from __future__ import annotations

import re
from io import BytesIO
from xml.sax.saxutils import escape

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..models_db import ClaimRow, Paper, Report
from ..report.report_chain import generate_report
from .routes_library import claim_json, get_session
from .routes_topics import _topic_or_404

router = APIRouter()


def report_json(session: Session, report: Report, label: str) -> dict:
    ids = report.cited_claim_ids or []
    rows = session.execute(
        select(ClaimRow, Paper.title, Paper.year).join(Paper, Paper.id == ClaimRow.paper_id).where(ClaimRow.claim_id.in_(ids))
    ).all() if ids else []
    return {
        "id": report.id,
        "topic_id": report.topic_id,
        "topic_label": label,
        "text": report.text,
        "cited_claim_ids": ids,
        "removed_count": len(report.removed_sentences or []),
        "model": report.model,
        "prompt_version": report.prompt_version,
        "created_at": report.created_at.isoformat(),
        "claims": {row.claim_id: {**claim_json(row), "paper_title": title, "year": year} for row, title, year in rows},
    }


@router.get("/topics/{topic_id}/report")
def topic_report(topic_id: int, regenerate: bool = False, session: Session = Depends(get_session)) -> dict:
    topic = _topic_or_404(session, topic_id)
    report = generate_report(session, topic, regenerate=regenerate)
    return report_json(session, report, topic.label)


def _numbered(session: Session, report: Report) -> tuple[str, list[str]]:
    """Replace claim ids with reference numbers and build the list of sources."""
    order: list[str] = []

    def replace(match: re.Match) -> str:
        claim_id = match.group(1)
        if claim_id not in order:
            order.append(claim_id)
        return f"[{order.index(claim_id) + 1}]"

    text = re.sub(r"\[([A-Za-z0-9]+_c\d+)\]", replace, report.text)
    sources = []
    for number, claim_id in enumerate(order, start=1):
        row = session.get(ClaimRow, claim_id)
        paper = session.get(Paper, row.paper_id) if row else None
        where = f"{paper.title if paper else claim_id}" + (f" ({paper.year})" if paper and paper.year else "")
        page = f", p. {row.page}" if row and row.page else ""
        quote = f': "{row.quote}"' if row else ""
        sources.append(f"[{number}] {where}{page}{quote}")
    return text, sources


@router.get("/topics/{topic_id}/report.md", response_class=PlainTextResponse)
def report_markdown(topic_id: int, session: Session = Depends(get_session)) -> PlainTextResponse:
    topic = _topic_or_404(session, topic_id)
    report = generate_report(session, topic)
    text, sources = _numbered(session, report)
    body = f"# {topic.label}\n\n{text}\n\n## Sources\n\n" + "\n\n".join(sources) + "\n"
    return PlainTextResponse(
        body, media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="report-{topic_id}.md"'}
    )


@router.get("/topics/{topic_id}/report.pdf")
def report_pdf(topic_id: int, session: Session = Depends(get_session)) -> Response:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    topic = _topic_or_404(session, topic_id)
    report = generate_report(session, topic)
    if report is None:
        raise HTTPException(status_code=404, detail="No report.")
    text, sources = _numbered(session, report)
    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=11, leading=16, spaceAfter=8)
    heading = ParagraphStyle("heading", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceBefore=12, spaceAfter=6)
    title = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=12)
    small = ParagraphStyle("small", fontName="Times-Roman", fontSize=9, leading=13, spaceAfter=6)

    story = [Paragraph(escape(topic.label), title)]
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 2 * mm))
        elif stripped.startswith("#"):
            story.append(Paragraph(escape(stripped.lstrip("# ")), heading))
        else:
            story.append(Paragraph(escape(stripped), body))
    story.append(Paragraph("Sources", heading))
    story.extend(Paragraph(escape(source), small) for source in sources)

    buffer = BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=25 * mm, rightMargin=25 * mm, topMargin=22 * mm, bottomMargin=22 * mm, title=topic.label).build(story)
    return Response(
        buffer.getvalue(), media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="report-{topic_id}.pdf"'}
    )
