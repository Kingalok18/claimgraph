"""One claim pair: its evidence, and the user's confirmation or correction."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .routes_labels import save_label
from .routes_library import get_session
from .routes_topics import get_pair_or_404, pair_json
from .schemas import PairLabelRequest

router = APIRouter()


@router.get("/pairs/{pair_id}")
def pair_detail(pair_id: int, session: Session = Depends(get_session)) -> dict:
    return pair_json(session, get_pair_or_404(session, pair_id))


@router.post("/pairs/{pair_id}/label")
def pair_label(pair_id: int, request: PairLabelRequest, session: Session = Depends(get_session)) -> dict:
    pair = get_pair_or_404(session, pair_id)
    save_label(
        session,
        "pair",
        pair.pair_key,
        {
            "relation": request.relation,
            "note": request.note,
            "predicted": pair.relation,
            "claim_a": pair.claim_a,
            "claim_b": pair.claim_b,
        },
    )
    return pair_json(session, pair)
