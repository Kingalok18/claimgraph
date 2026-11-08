"""Extraction precision from the user's claim checks. Waits until claims have been labelled."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from ..db import init_db, session_scope
from ..models_db import ClaimRow, Label, Paper
from .common import save_result, waiting

MIN_LABELS = 20


def main() -> Path:
    init_db()
    with session_scope() as session:
        rows = session.scalars(select(Label).where(Label.kind == "extraction").order_by(Label.created_at, Label.id)).all()
        latest = {row.target_id: row.value.get("verdict") for row in rows}
        proposed_dropped = sum(paper.claims_dropped or 0 for paper in session.scalars(select(Paper)))
        stored = session.query(ClaimRow).count()
    verification = {
        "claims_stored": stored,
        "claims_dropped_by_checks": proposed_dropped,
        "drop_rate": round(proposed_dropped / (stored + proposed_dropped), 4) if (stored + proposed_dropped) else None,
    }
    if len(latest) < MIN_LABELS:
        return save_result(
            "extraction",
            {
                "status": "waiting_for_labels",
                "reason": f"{len(latest)} of at least {MIN_LABELS} claims labelled. Use the Labelling screen.",
                "quote_check": verification,
            },
        )
    correct = sum(1 for verdict in latest.values() if verdict == "correct")
    return save_result(
        "extraction",
        {"status": "measured", "n": len(latest), "precision": round(correct / len(latest), 4), "quote_check": verification},
    )


if __name__ == "__main__":
    print(main())
