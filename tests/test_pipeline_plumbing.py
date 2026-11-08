"""The whole pipeline with stand-in models: storage, resume, and caching. Real models: test_pipeline_fixtures."""
from sqlalchemy import func, select

from fakes import fake_extract_chain
from claimgraph.db import session_scope
from claimgraph.eval.fixture_eval import compare_with_expected, read_outcome
from claimgraph.ingest.library import scan_library
from claimgraph.models_db import ClaimPair, ClaimRow, Label, Paper, Topic
from claimgraph.pipeline import run_analysis, run_question
from langchain_core.runnables import RunnableLambda

LABEL = RunnableLambda(lambda _: type("M", (), {"content": "Dropout and accuracy"})())


def label_chain():
    return RunnableLambda(lambda _: "Dropout and test accuracy")


def run(**kwargs):
    return run_analysis(extract_chain=fake_extract_chain(), label_chain=label_chain(), **kwargs)


def test_planted_relations_with_stand_in_models(fixture_papers, session, fake_models, expected):
    scan_library(session, fixture_papers)
    summary = run()
    assert summary["extraction"]["kept"] == 4
    papers, claims, pairs = read_outcome()
    outcome = compare_with_expected(expected, papers, claims, pairs)
    assert outcome["claims_found"] == 4
    assert outcome["relations_correct"] == outcome["relations_expected"], outcome["relations"]
    topic = session.scalar(select(Topic))
    assert topic.label == "Dropout and test accuracy"


def test_second_run_does_no_model_work(fixture_papers, session, fake_models):
    scan_library(session, fixture_papers)
    run()
    second = run()
    assert second["extraction"]["papers"] == 0
    assert second["indexed"] == 0 and second["topics"]["topics_new"] == 0
    assert second["relations"]["pairs_scored"] == 0 and second["labelled"] == 0


def test_limit_processes_only_that_many_papers(fixture_papers, session, fake_models):
    scan_library(session, fixture_papers)
    assert run(limit=2)["extraction"]["papers"] == 2
    session.expire_all()
    assert session.scalar(select(func.count()).select_from(Paper).where(Paper.extract_status == "pending")) == 2
    assert run()["extraction"]["papers"] == 2


def test_adding_papers_reuses_earlier_relations_and_keeps_labels(fixture_papers, session, fake_models, monkeypatch):
    from claimgraph import pipeline

    scan_library(session, fixture_papers)
    run(limit=2)
    session.expire_all()
    first_pair = session.scalar(select(ClaimPair))
    key = first_pair.pair_key
    session.add(Label(kind="pair", target_id=key, value={"relation": "unclear"}))
    session.commit()

    scored = []
    real = pipeline.get_nli_model()

    class Counting:
        batch_size = 8

        def score_pairs(self, pairs):
            scored.extend(pairs)
            return real.score_pairs(pairs)

    monkeypatch.setattr(pipeline, "get_nli_model", lambda: Counting())
    run()
    session.expire_all()
    # The topic grew, yet the earlier pair kept its result and was not scored again.
    assert session.scalar(select(ClaimPair).where(ClaimPair.pair_key == key)) is not None
    assert len(scored) == session.scalar(select(func.count(func.distinct(ClaimPair.pair_key)))) - 1
    assert session.scalar(select(func.count()).select_from(Label)) == 1


def test_llm_compare_is_only_used_for_conflicts(fixture_papers, session, fake_models):
    from claimgraph.relations.compare_chain import ConditionComparison

    calls = []

    def compare(_):
        calls.append(1)
        return ConditionComparison(conditions_match="unknown")

    scan_library(session, fixture_papers)
    run_analysis(extract_chain=fake_extract_chain(), label_chain=label_chain(), compare_chain=RunnableLambda(compare))
    # Conditions in the planted claims settle every pair by rules, so the slow model is never called.
    assert calls == []


def test_cancelled_run_keeps_its_work(fixture_papers, session, fake_models):
    import pytest

    from claimgraph.pipeline import Cancelled

    scan_library(session, fixture_papers)
    ticks = {"count": 0}

    def stop_later():
        ticks["count"] += 1
        return ticks["count"] > 5

    with pytest.raises(Cancelled):
        run(should_cancel=stop_later)
    session.expire_all()
    kept = session.scalar(select(func.count()).select_from(ClaimRow))
    assert kept >= 1
    summary = run()
    assert summary["extraction"]["kept"] + kept == 4


def test_question_topic(fixture_papers, session, fake_models, monkeypatch):
    from claimgraph.config import get_settings

    scan_library(session, fixture_papers)
    run()
    monkeypatch.setitem(get_settings().pipeline["topics"], "question_similarity", 0.1)
    result = run_question("Does dropout change test accuracy?")
    assert result["topic_id"] and result["claims"] == 4
    with session_scope() as db:
        topic = db.get(Topic, result["topic_id"])
        assert topic.source == "question" and topic.question_text.startswith("Does dropout")
        assert len(topic.pairs) > 0 and all(pair.relation for pair in topic.pairs)


def test_question_with_no_close_claims_says_so(fixture_papers, session, fake_models, monkeypatch):
    from claimgraph.config import get_settings

    scan_library(session, fixture_papers)
    run()
    monkeypatch.setitem(get_settings().pipeline["topics"], "question_similarity", 0.999)
    result = run_question("Something about the migration of arctic birds")
    assert result["topic_id"] is None and "No claims" in result["message"]


def test_collecting_tests_leaves_the_real_models_in_place():
    """A helper named *_test.py was once imported during collection and silently replaced the real models."""
    from claimgraph import pipeline

    assert pipeline.get_nli_model.__module__ == "claimgraph.relations.nli_infer"
    assert pipeline.build_extract_chain.__module__ == "claimgraph.claims.extract_chain"
    assert pipeline.build_label_chain.__module__ == "claimgraph.topics.label_chain"
