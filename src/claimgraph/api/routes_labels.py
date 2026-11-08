"""Hand labels for evaluation: claim checks and claim-pair relations."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..models_db import ClaimPair, ClaimRow, Label, Paper
from .routes_library import claim_json, get_session
from .schemas import ClaimLabelRequest

router = APIRouter()
_PRIORITY = {"contradicts": 0, "different_setting": 1, "supports": 2, "unclear": 3}


def save_label(session: Session, kind: str, target_id: str, value: dict) -> Label:
    """Store a label in the database, and keep a plain-text copy in data/labels."""
    label = Label(kind=kind, target_id=target_id, value=value)
    session.add(label)
    session.commit()
    labels_dir = get_settings().labels_dir
    labels_dir.mkdir(parents=True, exist_ok=True)
    with (labels_dir / f"{kind}_labels.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"target_id": target_id, **value}, ensure_ascii=False) + "\n")
    return label


@router.get("/labels/stats")
def label_stats(session: Session = Depends(get_session)) -> dict:
    return {
        "pair_labels": session.scalar(select(func.count(func.distinct(Label.target_id))).where(Label.kind == "pair")) or 0,
        "claim_labels": session.scalar(select(func.count(func.distinct(Label.target_id))).where(Label.kind == "extraction")) or 0,
        "pairs_total": session.scalar(select(func.count(func.distinct(ClaimPair.pair_key)))) or 0,
        "claims_total": session.scalar(select(func.count()).select_from(ClaimRow)) or 0,
    }


@router.get("/labels/queue")
def label_queue(kind: str = Query(default="pair", pattern="^(pair|extraction)$"), limit: int = Query(default=20, ge=1, le=100), session: Session = Depends(get_session)) -> dict:
    """Items not labelled yet. Pairs come conflicts first: they carry the research question."""
    done = set(session.scalars(select(Label.target_id).where(Label.kind == kind)).all())
    if kind == "pair":
        pairs = session.scalars(select(ClaimPair).where(ClaimPair.relation.is_not(None))).all()
        seen: set[str] = set()
        chosen = []
        for pair in sorted(pairs, key=lambda item: (_PRIORITY.get(item.relation, 9), item.id)):
            if pair.pair_key in done or pair.pair_key in seen:
                continue
            seen.add(pair.pair_key)
            chosen.append(pair)
            if len(chosen) >= limit:
                break
        return {"items": [{"id": pair.id} for pair in chosen]}

    rows = session.scalars(select(ClaimRow).order_by(ClaimRow.claim_id)).all()
    chosen_claims = [row for row in rows if row.claim_id not in done][:limit]
    titles = {paper_id: title for paper_id, title in session.execute(select(Paper.id, Paper.title)).all()}
    return {"items": [{**claim_json(row), "paper_title": titles.get(row.paper_id, "")} for row in chosen_claims]}


@router.post("/labels/claim")
def label_claim(request: ClaimLabelRequest, session: Session = Depends(get_session)) -> dict:
    if session.get(ClaimRow, request.claim_id) is None:
        raise HTTPException(status_code=404, detail="Claim not found.")
    save_label(session, "extraction", request.claim_id, {"verdict": request.verdict, "note": request.note})
    return {"ok": True}
