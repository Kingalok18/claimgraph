"""The analysis pipeline: claims -> index -> topics -> pairs -> relations -> labels.

Models run one after another and are released in between, so that a machine with 8 GB of RAM
never holds the language model, the embedding model and the NLI model at the same time.
Every step keeps its progress in the database, so a stopped run continues where it left off.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .claims.extract_chain import build_extract_chain, extract_paper
from .claims.schema import Conditions
from .config import get_settings
from .db import session_scope
from .embed.embeddings import release_embeddings
from .embed.vectorstore import get_vectors, index_claims
from .llm import unload_llm
from .models_db import ClaimPair, ClaimRow, ClaimTopic, Paper, Topic
from .relations.compare_chain import build_compare_chain, compare_conditions
from .relations.decide import OPPOSITE, decide
from .relations.nli_infer import get_nli_model, release_nli_model
from .topics.group import cluster_vectors, generate_pairs, pair_key, topic_key
from .topics.label_chain import build_label_chain, label_topic
from .topics.question_filter import claims_for_question

log = logging.getLogger(__name__)
Progress = Callable[..., None]


class Cancelled(Exception):
    pass


def _noop(*_args, **_kwargs) -> None:
    pass


def _claim_dict(row: ClaimRow) -> dict:
    return {
        "claim_id": row.claim_id,
        "paper_id": row.paper_id,
        "text": row.text,
        "intervention": row.intervention,
        "outcome": row.outcome,
        "direction": row.direction,
        "conditions": row.conditions or {},
    }


def verified_claims(session: Session) -> list[ClaimRow]:
    return list(session.scalars(select(ClaimRow).where(ClaimRow.quote_verified.is_(True)).order_by(ClaimRow.claim_id)))


def papers_to_extract(session: Session, limit: int | None = None, paper_ids: list[str] | None = None) -> list[Paper]:
    query = (
        select(Paper)
        .options(selectinload(Paper.sections), selectinload(Paper.claims))
        .where(Paper.parse_status == "parsed", Paper.extract_status == "pending", Paper.missing.is_(False))
        .order_by(Paper.added_at, Paper.id)
    )
    if paper_ids:
        query = query.where(Paper.id.in_(paper_ids))
    if limit:
        query = query.limit(limit)
    return list(session.scalars(query))


def step_extract(session: Session, progress: Progress, should_cancel, limit=None, paper_ids=None, chain=None) -> dict:
    papers = papers_to_extract(session, limit, paper_ids)
    totals = {"papers": 0, "proposed": 0, "kept": 0, "dropped": 0, "chunks_failed": 0}
    if not papers:
        return totals
    chain = chain if chain is not None else build_extract_chain()
    for index, paper in enumerate(papers, start=1):
        if should_cancel():
            raise Cancelled()
        title = paper.title or paper.id
        progress("claims", f"Finding claims in paper {index} of {len(papers)}", index - 1, len(papers), title)
        stats = extract_paper(
            session,
            paper,
            chain=chain,
            on_chunk=lambda done, total, i=index, t=title: progress(
                "claims", f"Finding claims in paper {i} of {len(papers)}", i - 1, len(papers), f"{t} (part {done} of {total})"
            ),
            should_cancel=should_cancel,
        )
        if stats.get("cancelled"):
            raise Cancelled()
        totals["papers"] += 1
        for key in ("proposed", "kept", "dropped", "chunks_failed"):
            totals[key] += stats.get(key, 0)
    totals["drop_rate"] = round(totals["dropped"] / totals["proposed"], 4) if totals["proposed"] else 0.0
    log.info("Extraction totals: %s", totals)
    return totals


def step_index(session: Session, progress: Progress) -> int:
    claims = [_claim_dict(row) for row in verified_claims(session)]
    progress("index", "Preparing claims for comparison", 0, len(claims), None)
    added = index_claims(claims) if claims else 0
    release_embeddings()
    return added


def _pair_cache(session: Session) -> dict[str, ClaimPair]:
    """Relations already worked out, so that re-grouping topics does not repeat the model work."""
    rows = session.scalars(select(ClaimPair).where(ClaimPair.relation.is_not(None))).all()
    return {row.pair_key: row for row in rows}


def _copy_of(row: ClaimPair) -> dict:
    return {
        "nli_label": row.nli_label,
        "nli_prob": row.nli_prob,
        "conditions_match": row.conditions_match,
        "condition_diff": row.condition_diff,
        "relation": row.relation,
    }


def step_topics(session: Session, progress: Progress) -> dict:
    """Cluster all claims. A topic whose claims have not changed is kept with its label and report."""
    config = get_settings().pipeline["topics"]
    rows = verified_claims(session)
    claim_ids = [row.claim_id for row in rows]
    vectors = get_vectors(claim_ids)
    usable = [claim_id for claim_id in claim_ids if claim_id in vectors]
    progress("topics", "Grouping claims into topics", 0, len(usable), None)

    matrix = np.array([vectors[claim_id] for claim_id in usable]) if usable else np.zeros((0, 1))
    labels = cluster_vectors(
        matrix, int(config["min_cluster_size"]), float(config["attach_noise_similarity"]), float(config["merge_similarity"])
    )

    clusters: dict[int, list[int]] = {}
    for position, label in enumerate(labels):
        if label != -1:
            clusters.setdefault(int(label), []).append(position)

    existing = {topic.topic_key: topic for topic in session.scalars(select(Topic).where(Topic.source == "auto"))}
    wanted: set[str] = set()
    created = 0
    for positions in clusters.values():
        members = [usable[position] for position in positions]
        key = topic_key(members)
        wanted.add(key)
        if key in existing:
            continue
        centroid = matrix[positions].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        topic = Topic(topic_key=key, label="Untitled topic", source="auto")
        session.add(topic)
        session.flush()
        for position in positions:
            unit = matrix[position] / (np.linalg.norm(matrix[position]) or 1.0)
            session.add(ClaimTopic(claim_id=usable[position], topic_id=topic.id, score=float(unit @ centroid)))
        created += 1
    return_cache = {key: _copy_of(row) for key, row in _pair_cache(session).items()}
    for key, topic in existing.items():
        if key not in wanted:
            session.delete(topic)
    session.commit()
    return {"topics_total": len(wanted), "topics_new": created, "pair_cache": return_cache}


def _topic_claims(session: Session, topic: Topic) -> list[ClaimRow]:
    ids = [link.claim_id for link in topic.claim_links]
    if not ids:
        return []
    return list(session.scalars(select(ClaimRow).where(ClaimRow.claim_id.in_(ids)).order_by(ClaimRow.claim_id)))


def create_pairs(session: Session, topic: Topic, cache: dict[str, dict]) -> int:
    """Make the claim pairs of a topic that has none yet. Known relations are reused from the cache."""
    if topic.pairs:
        return 0
    neighbours = int(get_settings().pipeline["pairs"]["neighbours_per_claim"])
    claims = _topic_claims(session, topic)
    ids = [claim.claim_id for claim in claims]
    vectors = get_vectors(ids)
    ids = [claim_id for claim_id in ids if claim_id in vectors]
    paper_of = {claim.claim_id: claim.paper_id for claim in claims}
    if len(ids) < 2:
        return 0
    matrix = np.array([vectors[claim_id] for claim_id in ids])
    count = 0
    for claim_a, claim_b in generate_pairs(ids, matrix, paper_of, neighbours):
        key = pair_key(claim_a, claim_b)
        session.add(ClaimPair(topic_id=topic.id, pair_key=key, claim_a=claim_a, claim_b=claim_b, **cache.get(key, {})))
        count += 1
    session.commit()
    return count


def step_relations(session: Session, progress: Progress, should_cancel, compare_chain=None) -> dict:
    """NLI for all unscored pairs, then the conditions check, then the decision rule."""
    settings = get_settings()
    thresholds = settings.pipeline["relations"]
    support_th = float(thresholds["nli_support_threshold"])
    contradict_th = float(thresholds["nli_contradict_threshold"])
    max_llm = int(thresholds.get("max_llm_compares", 200))

    pending = list(session.scalars(select(ClaimPair).where(ClaimPair.relation.is_(None))))
    if not pending:
        return {"pairs_scored": 0, "llm_compares": 0}
    claims = {row.claim_id: row for row in verified_claims(session)}
    # Group identical pairs across topics so that the models see each one once.
    by_key: dict[str, list[ClaimPair]] = {}
    for row in pending:
        by_key.setdefault(row.pair_key, []).append(row)
    unique = list(by_key.values())

    # NLI first, in batches, then release the model before the language model may load.
    nli = get_nli_model()
    batch = max(8, nli.batch_size)
    results = []
    for start in range(0, len(unique), batch):
        if should_cancel():
            raise Cancelled()
        progress("relations", "Comparing claims", start, len(unique), None)
        part = unique[start : start + batch]
        results.extend(nli.score_pairs([(claims[rows[0].claim_a].text, claims[rows[0].claim_b].text) for rows in part]))
    release_nli_model()

    llm_used = 0
    chain = compare_chain
    for index, (rows, result) in enumerate(zip(unique, results)):
        if should_cancel():
            raise Cancelled()
        progress("relations", "Checking study conditions", index, len(unique), None)
        first, second = claims[rows[0].claim_a], claims[rows[0].claim_b]
        # The conditions only change the outcome for conflicts and opposite directions. Elsewhere the
        # LLM's answer could not alter the relation, so its slow call is skipped.
        conflict = (result.label == "contradiction" and result.prob >= contradict_th) or (
            (first.direction, second.direction) in OPPOSITE
        )
        use_llm = conflict and llm_used < max_llm
        if use_llm and chain is None:
            chain = build_compare_chain()
        comparison = compare_conditions(
            first.text,
            Conditions(**(first.conditions or {})),
            second.text,
            Conditions(**(second.conditions or {})),
            chain=chain,
            use_llm=use_llm,
        )
        if comparison["decided_by"] == "llm":
            llm_used += 1
        relation = decide(
            result.label,
            result.prob,
            comparison["conditions_match"],
            first.direction,
            second.direction,
            support_th=support_th,
            contradict_th=contradict_th,
        )
        for row in rows:
            row.nli_label = result.label
            row.nli_prob = result.prob
            row.conditions_match = comparison["conditions_match"]
            row.condition_diff = {**comparison, "nli_probs": result.probs}
            row.relation = relation
        session.commit()
    return {"pairs_scored": len(unique), "llm_compares": llm_used}


def step_labels(session: Session, progress: Progress, should_cancel, chain=None) -> int:
    topics = list(session.scalars(select(Topic).where(Topic.source == "auto", Topic.label == "Untitled topic")))
    if not topics:
        return 0
    chain = chain if chain is not None else build_label_chain()
    for index, topic in enumerate(topics):
        if should_cancel():
            raise Cancelled()
        progress("labels", "Naming topics", index, len(topics), None)
        claims = [_claim_dict(row) for row in _topic_claims(session, topic)]
        topic.label = label_topic(claims, chain=chain)[:255] if claims else "Empty topic"
        session.commit()
    return len(topics)


def run_analysis(
    limit: int | None = None,
    paper_ids: list[str] | None = None,
    progress: Progress = _noop,
    should_cancel=lambda: False,
    extract_chain=None,
    compare_chain=None,
    label_chain=None,
) -> dict:
    """Run every step. Safe to call again: finished work is skipped."""
    summary: dict = {}
    with session_scope() as session:
        summary["extraction"] = step_extract(session, progress, should_cancel, limit, paper_ids, extract_chain)
        unload_llm()

        summary["indexed"] = step_index(session, progress)
        topics_info = step_topics(session, progress)
        cache = topics_info.pop("pair_cache")
        summary["topics"] = topics_info

        topics = list(session.scalars(select(Topic).options(selectinload(Topic.pairs), selectinload(Topic.claim_links))))
        made = 0
        for topic in topics:
            made += create_pairs(session, topic, cache)
        summary["pairs_created"] = made

        summary["relations"] = step_relations(session, progress, should_cancel, compare_chain)
        summary["labelled"] = step_labels(session, progress, should_cancel, label_chain)
        unload_llm()
    return summary


def run_question(question: str, progress: Progress = _noop, should_cancel=lambda: False, compare_chain=None) -> dict:
    """Create a topic from a research question, then work out its relations."""
    question = question.strip()
    with session_scope() as session:
        progress("question", "Finding claims for your question", 0, 1, None)
        unload_llm()
        hits = claims_for_question(question)
        release_embeddings()
        if not hits:
            return {"topic_id": None, "claims": 0, "message": "No claims are close enough to this question yet."}
        ids = [claim_id for claim_id, _ in hits]
        key = topic_key(ids, question)
        topic = session.scalar(select(Topic).where(Topic.topic_key == key))
        if topic is None:
            topic = Topic(topic_key=key, label=question[:255], source="question", question_text=question)
            session.add(topic)
            session.flush()
            for claim_id, score in hits:
                session.add(ClaimTopic(claim_id=claim_id, topic_id=topic.id, score=score))
            session.commit()
            session.refresh(topic)
        cache = {pair: _copy_of(row) for pair, row in _pair_cache(session).items()}
        pairs = create_pairs(session, topic, cache)
        relations = step_relations(session, progress, should_cancel, compare_chain)
        unload_llm()
        return {"topic_id": topic.id, "claims": len(ids), "pairs": pairs, "relations": relations}
