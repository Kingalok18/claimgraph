"""Topics and their claim graphs."""
from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ..graph.build import build_graph
from ..graph.export import to_cytoscape
from ..models_db import ClaimPair, ClaimRow, Label, Paper, Topic
from ..pipeline import run_question
from . import jobs
from .routes_library import get_session
from .schemas import QuestionRequest
from ..relations.compare_chain import reason_for

router = APIRouter()


def corrected_relations(session: Session, pair_keys: list[str]) -> dict[str, str]:
    """The user's latest label for each pair. A correction overrides the computed relation everywhere."""
    if not pair_keys:
        return {}
    rows = session.scalars(
        select(Label).where(Label.kind == "pair", Label.target_id.in_(pair_keys)).order_by(Label.created_at, Label.id)
    ).all()
    return {row.target_id: row.value.get("relation") for row in rows}


@router.get("/topics")
def topics(session: Session = Depends(get_session)) -> dict:
    rows = session.scalars(select(Topic).options(selectinload(Topic.claim_links), selectinload(Topic.pairs))).all()
    corrections = corrected_relations(session, [pair.pair_key for topic in rows for pair in topic.pairs])
    output = []
    for topic in rows:
        counts = Counter(corrections.get(pair.pair_key) or pair.relation or "unclear" for pair in topic.pairs)
        output.append(
            {
                "id": topic.id,
                "label": topic.label,
                "source": topic.source,
                "question_text": topic.question_text,
                "claims": len(topic.claim_links),
                "pairs": len(topic.pairs),
                "supports": counts.get("supports", 0),
                "contradicts": counts.get("contradicts", 0),
                "different_setting": counts.get("different_setting", 0),
                "unclear": counts.get("unclear", 0),
            }
        )
    output.sort(key=lambda item: (-item["contradicts"], -item["pairs"], item["label"]))
    return {"topics": output}


@router.post("/topics/question")
def topic_question(request: QuestionRequest) -> dict:
    def work(context: jobs.JobContext) -> dict:
        return run_question(request.question, progress=context.progress, should_cancel=context.should_cancel)

    return {"job_id": jobs.submit("question", work)}


def _topic_or_404(session: Session, topic_id: int) -> Topic:
    topic = session.scalar(
        select(Topic).options(selectinload(Topic.claim_links), selectinload(Topic.pairs)).where(Topic.id == topic_id)
    )
    if topic is None:
        raise HTTPException(status_code=404, detail="Topic not found.")
    return topic


@router.get("/topics/{topic_id}/graph")
def topic_graph(topic_id: int, session: Session = Depends(get_session)) -> dict:
    topic = _topic_or_404(session, topic_id)
    ids = [link.claim_id for link in topic.claim_links]
    rows = session.execute(
        select(ClaimRow, Paper.title, Paper.year).join(Paper, Paper.id == ClaimRow.paper_id).where(ClaimRow.claim_id.in_(ids))
    ).all() if ids else []
    claims = [
        {"claim_id": row.claim_id, "paper_id": row.paper_id, "text": row.text, "paper_title": title or "", "year": year}
        for row, title, year in rows
    ]
    corrections = corrected_relations(session, [pair.pair_key for pair in topic.pairs])
    pairs = [
        {
            "id": pair.id,
            "claim_a": pair.claim_a,
            "claim_b": pair.claim_b,
            "relation": corrections.get(pair.pair_key) or pair.relation,
        }
        for pair in topic.pairs
    ]
    return {
        "topic": {"id": topic.id, "label": topic.label, "source": topic.source, "question_text": topic.question_text},
        **to_cytoscape(build_graph(claims, pairs)),
    }


@router.get("/topics/{topic_id}/pairs")
def topic_pairs(topic_id: int, session: Session = Depends(get_session)) -> dict:
    topic = _topic_or_404(session, topic_id)
    return {"pairs": [{"id": pair.id, "claim_a": pair.claim_a, "claim_b": pair.claim_b, "relation": pair.relation} for pair in topic.pairs]}


def pair_json(session: Session, pair: ClaimPair) -> dict:
    first = session.get(ClaimRow, pair.claim_a)
    second = session.get(ClaimRow, pair.claim_b)

    def side(claim: ClaimRow) -> dict:
        paper = session.get(Paper, claim.paper_id)
        return {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "quote": claim.quote,
            "page": claim.page,
            "direction": claim.direction,
            "conditions": claim.conditions or {},
            "hedged": claim.hedged,
            "paper_id": claim.paper_id,
            "paper_title": paper.title if paper else "",
            "year": paper.year if paper else None,
        }

    label = session.scalar(
        select(Label).where(Label.kind == "pair", Label.target_id == pair.pair_key).order_by(Label.created_at.desc(), Label.id.desc())
    )
    diff = pair.condition_diff or {}
    return {
        "id": pair.id,
        "topic_id": pair.topic_id,
        "pair_key": pair.pair_key,
        "a": side(first),
        "b": side(second),
        "nli_label": pair.nli_label,
        "nli_prob": pair.nli_prob,
        "conditions_match": pair.conditions_match,
        "differing_fields": diff.get("differing_fields", []),
        "field_status": diff.get("field_status", {}),
        "reason": reason_for(pair.condition_diff),
        "decided_by": diff.get("decided_by"),
        "relation": pair.relation,
        "user_label": label.value.get("relation") if label else None,
    }


def get_pair_or_404(session: Session, pair_id: int) -> ClaimPair:
    pair = session.get(ClaimPair, pair_id)
    if pair is None:
        raise HTTPException(status_code=404, detail="Pair not found.")
    return pair
