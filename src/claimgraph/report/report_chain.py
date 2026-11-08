"""The evidence map. Every sentence that survives cites existing claims; see citation_check."""
from __future__ import annotations

import json
import logging
import threading

from langchain_core.output_parsers import StrOutputParser
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..claims.extract_chain import load_prompt
from ..config import get_settings
from ..llm import get_chat_model
from ..models_db import ClaimPair, ClaimRow, Report, Topic
from ..relations.compare_chain import reason_for
from .citation_check import check_citations
from .grounding import Grounded, compose_report, ground_sentences

log = logging.getLogger(__name__)
PROMPT_VERSION = "write_report.v1"
_generation_lock = threading.Lock()
_ORDER = {"contradicts": 0, "different_setting": 1, "supports": 2, "unclear": 3}
MAX_UNCLEAR_FOR_MODEL = 5


def build_report_chain(llm=None):
    model = llm if llm is not None else get_chat_model("main")
    chain = load_prompt(PROMPT_VERSION) | model | StrOutputParser()
    return chain.with_config(metadata={"chain_name": "write_report", "prompt_version": PROMPT_VERSION})


def template_report(claims: list[dict], relations: list[dict]) -> str:
    """A plain evidence map from the stored relations alone."""
    return compose_report(claims, relations)


def _for_the_model(claims: list[dict], relations: list[dict]) -> tuple[list[dict], list[dict]]:
    """What the model may phrase. Settings are written from stored conditions, so it does not see those pairs."""
    chosen = [pair for pair in relations if pair["relation"] in ("supports", "contradicts")]
    chosen += [pair for pair in relations if pair["relation"] == "unclear"][:MAX_UNCLEAR_FOR_MODEL]
    named = {claim_id for pair in chosen for claim_id in (pair["claim_a"], pair["claim_b"])}
    return [claim for claim in claims if claim["claim_id"] in named], chosen


def _inputs(session: Session, topic: Topic) -> tuple[list[dict], list[dict]]:
    limits = get_settings().pipeline["report"]
    ids = [link.claim_id for link in sorted(topic.claim_links, key=lambda link: -link.score)]
    pairs = sorted(topic.pairs, key=lambda pair: _ORDER.get(pair.relation or "unclear", 9))[: int(limits["max_relations"])]
    # Claims named by the chosen relations come first, so every relation stays citable.
    chosen: list[str] = []
    for pair in pairs:
        for claim_id in (pair.claim_a, pair.claim_b):
            if claim_id not in chosen:
                chosen.append(claim_id)
    for claim_id in ids:
        if claim_id not in chosen:
            chosen.append(claim_id)
    chosen = chosen[: int(limits["max_claims"])]
    rows = session.scalars(select(ClaimRow).where(ClaimRow.claim_id.in_(chosen))).all() if chosen else []
    claims = [
        {
            "claim_id": row.claim_id,
            "text": row.text,
            "direction": row.direction,
            "conditions": {key: value for key, value in (row.conditions or {}).items() if value},
        }
        for row in sorted(rows, key=lambda item: chosen.index(item.claim_id))
    ]
    relations = [
        {
            "claim_a": pair.claim_a,
            "claim_b": pair.claim_b,
            "relation": pair.relation or "unclear",
            "reason": reason_for(pair.condition_diff),
            "differing_fields": (pair.condition_diff or {}).get("differing_fields") or [],
        }
        for pair in pairs
        if pair.claim_a in chosen and pair.claim_b in chosen
    ]
    return claims, relations


def latest_report(session: Session, topic_id: int) -> Report | None:
    return session.scalar(select(Report).where(Report.topic_id == topic_id).order_by(Report.created_at.desc(), Report.id.desc()))


def generate_report(session: Session, topic: Topic, chain=None, regenerate: bool = False) -> Report:
    """Fetch the stored report or write a new one. Only one report is written at a time."""
    with _generation_lock:
        if not regenerate:
            existing = latest_report(session, topic.id)
            if existing is not None:
                return existing

        claims, relations = _inputs(session, topic)
        known = {claim["claim_id"] for claim in claims}
        config = get_settings().models["llm"]
        model_name = str(config.get("gguf_file") or config.get("model") or config.get("hf_model_id"))
        removed: list[dict] = []
        grounded = Grounded()
        prompt_claims, prompt_relations = _for_the_model(claims, relations)
        # With no agreement or conflict to phrase, a slow model call could add only risk.
        if any(pair["relation"] in ("supports", "contradicts") for pair in prompt_relations):
            try:
                chain = chain if chain is not None else build_report_chain()
                raw = chain.invoke(
                    {
                        "topic": topic.label,
                        "claims": json.dumps(prompt_claims, ensure_ascii=False),
                        "relations": json.dumps(prompt_relations, ensure_ascii=False),
                    }
                )
                checked = check_citations(raw, known)
                removed.extend(checked.removed)
                grounded = ground_sentences(checked.text, claims, relations)
                removed.extend(grounded.removed)
            except Exception as exc:  # noqa: BLE001
                log.warning("Report generation failed for topic %s: %s", topic.id, exc)

        used_model = any(grounded.sentences.values())
        final = check_citations(compose_report(claims, relations, grounded), known)
        text, cited = final.text, final.cited
        removed.extend(final.removed)
        prompt_version = PROMPT_VERSION if used_model else "template"
        if not used_model:
            model_name = "template"

        report = Report(
            topic_id=topic.id,
            text=text,
            cited_claim_ids=cited,
            removed_sentences=removed,
            prompt_version=prompt_version,
            model=model_name[:128],
        )
        session.add(report)
        session.commit()
        return report


def pairs_of(session: Session, topic_id: int) -> list[ClaimPair]:
    return list(session.scalars(select(ClaimPair).where(ClaimPair.topic_id == topic_id)))
