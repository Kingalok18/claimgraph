"""Relation accuracy against the user's claim-pair labels. Waits until pairs have been labelled."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import select

from ..db import init_db, session_scope
from ..models_db import Label
from .common import classification_metrics, save_result, waiting

RELATIONS = ["supports", "contradicts", "different_setting", "unclear"]
MIN_LABELS = 20


def main() -> Path:
    init_db()
    with session_scope() as session:
        rows = session.scalars(select(Label).where(Label.kind == "pair").order_by(Label.created_at, Label.id)).all()
        latest = {row.target_id: row.value for row in rows}
    if len(latest) < MIN_LABELS:
        return waiting("pairs", f"{len(latest)} of at least {MIN_LABELS} pairs labelled. Use the Labelling screen.")
    gold = [value["relation"] for value in latest.values()]
    predicted = [value.get("predicted") or "unclear" for value in latest.values()]
    return save_result("pairs", {"status": "measured", **classification_metrics(gold, predicted, RELATIONS)})


if __name__ == "__main__":
    print(main())
