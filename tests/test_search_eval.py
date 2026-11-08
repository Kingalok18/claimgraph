"""Optional online search (mocked network) and evaluation scripts."""
import json

import pytest
from sqlalchemy import func, select

from claimgraph.config import get_settings
from claimgraph.eval import common, extraction_eval, pairs_eval, report_eval
from claimgraph.eval.biodivergence_eval import nli_to_relation
from claimgraph.eval.fixture_eval import compare_with_expected
from claimgraph.eval.scifact_nli_eval import build_examples
from claimgraph.models_db import Label, Paper, Section
from claimgraph.search import arxiv_client, openalex, search_agent
from claimgraph.search.cache import cache_get, cache_put

FEED = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry><id>http://arxiv.org/abs/2401.00001</id><published>2024-01-02T00:00:00Z</published><title>Image  forgery detection</title>
<summary>We detect forgery in images.</summary><author><name>A. Author</name></author><author><name>B. Writer</name></author>
<link title="pdf" href="https://arxiv.org/pdf/2401.00001"/></entry></feed>"""


def test_cache_round_trip():
    assert cache_get("x", "k") is None
    cache_put("x", "k", {"a": 1})
    assert cache_get("x", "k") == {"a": 1}


def test_openalex_rebuilds_abstract_from_positions():
    assert openalex.abstract_from_index({"world": [1], "hello": [0]}) == "hello world"
    assert openalex.abstract_from_index(None) is None


def test_openalex_record():
    record = openalex.to_record(
        {
            "id": "https://openalex.org/W1", "title": "T", "publication_year": 2023, "doi": "https://doi.org/10.1/x",
            "authorships": [{"author": {"display_name": "A"}}], "cited_by_count": 5, "referenced_works": ["W2"],
        }
    )
    assert record["doi"] == "10.1/x" and record["authors"] == "A" and record["references"] == ["W2"]


def test_arxiv_feed_parses():
    record = arxiv_client.parse_feed(FEED)[0]
    assert record["title"] == "Image forgery detection" and record["year"] == 2024
    assert record["authors"] == "A. Author, B. Writer" and record["pdf_url"].endswith("2401.00001")


def _record(title, doi=None, abstract="About image forgery.", references=None):
    return {"title": title, "doi": doi, "abstract": abstract, "year": 2022, "authors": "A", "venue": "V", "citations": 0, "references": references or [], "origin": "openalex"}


def test_merge_removes_duplicates_by_doi_and_title():
    merged = search_agent.merge_records(
        [_record("First paper", doi="10.1/a")], [_record("Other title", doi="10.1/A"), _record("First-Paper!"), _record("New")]
    )
    assert [record["title"] for record in merged] == ["First paper", "First-Paper!", "New"]


def test_agent_stops_at_loop_limit_and_budget(monkeypatch):
    monkeypatch.setattr(search_agent, "search_works", lambda q, limit: [_record("Image forgery one", references=["W9"])])
    monkeypatch.setattr(search_agent, "search_arxiv", lambda q, limit: [_record("Image forgery two")])
    calls = []

    def fetch(ids):
        calls.append(ids)
        return [_record(f"Image forgery ref {len(calls)}", references=["W9"])]

    monkeypatch.setattr(search_agent, "fetch_by_ids", fetch)
    records = search_agent.run_search_agent("image forgery")
    assert 1 <= len(calls) <= get_settings().pipeline["online_search"]["max_loops"]
    assert len(records) >= 2


def test_agent_survives_a_failing_source(monkeypatch):
    import httpx

    def down(*_a, **_k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(search_agent, "search_works", down)
    monkeypatch.setattr(search_agent, "search_arxiv", lambda q, limit: [_record("Image forgery")])
    assert [record["title"] for record in search_agent.run_search_agent("image forgery")] == ["Image forgery"]


def test_online_papers_are_marked_and_not_duplicated(session):
    records = [_record("Paper one", doi="10.1/a"), _record("No abstract", abstract=None)]
    assert search_agent.add_online_papers(session, records, download=False) == 1
    assert search_agent.add_online_papers(session, records, download=False) == 0
    paper = session.scalar(select(Paper))
    assert paper.source == "online" and paper.extract_status == "pending"
    assert session.scalar(select(func.count()).select_from(Section)) == 1


def test_only_arxiv_pdfs_are_downloaded():
    assert search_agent._download_arxiv_pdf({"origin": "openalex", "pdf_url": "https://evil.example/a.pdf"}) is None
    assert search_agent._download_arxiv_pdf({"origin": "arxiv", "pdf_url": "https://evil.example/a.pdf"}) is None


def test_scifact_examples():
    corpus = [{"doc_id": 1, "abstract": ["S0.", "S1.", "S2."]}, {"doc_id": 2, "abstract": ["N0.", "N1."]}]
    claims = [
        {"claim": "C1", "evidence": {"1": [{"sentences": [1], "label": "SUPPORT"}, {"sentences": [0, 2], "label": "CONTRADICT"}]}, "cited_doc_ids": [1]},
        {"claim": "C2", "evidence": {}, "cited_doc_ids": [2]},
    ]
    examples = build_examples(claims, corpus)
    assert {"premise": "S1.", "hypothesis": "C1", "label": "entailment"} in examples
    assert {"premise": "S0. S2.", "hypothesis": "C1", "label": "contradiction"} in examples
    assert {"premise": "N0. N1.", "hypothesis": "C2", "label": "neutral"} in examples


def test_metrics_are_computed_not_assumed():
    metrics = common.classification_metrics(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
    assert metrics["n"] == 3 and metrics["accuracy"] == pytest.approx(0.6667, abs=1e-3)
    assert metrics["per_class"]["a"]["recall"] == 0.5


def test_nli_to_relation():
    assert nli_to_relation("entailment") == "supports" and nli_to_relation("neutral") == "unclear"


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_waiting_evaluations_hold_no_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "get_settings", lambda: type("S", (), {"results_dir": tmp_path})())
    path = pairs_eval.main()
    data = _read(path)
    assert data["status"] == "waiting_for_labels" and "accuracy" not in data
    assert "precision" not in _read(extraction_eval.main())
    assert _read(report_eval.main())["status"] == "waiting_for_reports"


def test_pairs_eval_measures_once_labelled(session, tmp_path, monkeypatch):
    monkeypatch.setattr(common, "get_settings", lambda: type("S", (), {"results_dir": tmp_path})())
    for index in range(pairs_eval.MIN_LABELS):
        session.add(Label(kind="pair", target_id=f"k{index}", value={"relation": "supports", "predicted": "supports" if index else "unclear"}))
    session.commit()
    data = _read(pairs_eval.main())
    assert data["status"] == "measured" and data["n"] == pairs_eval.MIN_LABELS
    assert data["accuracy"] == pytest.approx((pairs_eval.MIN_LABELS - 1) / pairs_eval.MIN_LABELS)


def test_expected_comparison_flags_wrong_and_missing(expected):
    files = {paper["file"]: paper["key"] for paper in expected["papers"]}
    claims = [
        {"claim_id": f"{paper['key']}_c1", "paper_id": paper["key"], "quote": paper["claims"][0]["quote"], "quote_verified": True}
        for paper in expected["papers"][:3]
    ]
    pairs = [{"claim_a": "A_c1", "claim_b": "B_c1", "relation": "contradicts", "nli_label": "contradiction", "nli_prob": 0.9, "conditions_match": "same"}]
    outcome = compare_with_expected(expected, files, claims, pairs)
    assert outcome["claims_found"] == 3 and outcome["planted"]["D1"] is None
    assert outcome["relations_correct"] == 0
    assert outcome["relations"][0]["got"] == "contradicts" and outcome["relations"][0]["ok"] is False


def test_results_table_uses_only_json(tmp_path, monkeypatch):
    import make_results_table as table

    monkeypatch.setattr(table, "RESULTS", tmp_path)
    (tmp_path / "scifact_nli.json").write_text(
        json.dumps({"name": "scifact_nli", "run_date": "D", "status": "measured", "n": 2, "accuracy": 0.5, "macro_f1": 0.4, "per_class": {}}),
        encoding="utf-8",
    )
    (tmp_path / "pairs.json").write_text(json.dumps({"name": "pairs", "status": "waiting_for_labels", "reason": "0 of 20"}), encoding="utf-8")
    text = table.main().read_text(encoding="utf-8")
    assert "accuracy 0.5" in text and "Run date: D" in text
    assert "waiting for labels" in text and "0 of 20" in text


def test_user_papers_summary_counts_what_is_stored(fixture_papers, session):
    from claimgraph.claims.extract_chain import extract_paper
    from claimgraph.eval.user_papers_eval import hash_files, pick_sample, summarise
    from claimgraph.ingest.library import scan_library
    from fakes import fake_extract_chain

    before = hash_files(fixture_papers)
    scan_library(session, fixture_papers)
    ids = pick_sample(session, ["fixture_a"])
    from claimgraph.models_db import Paper

    extract_paper(session, session.get(Paper, ids[0]), chain=fake_extract_chain(extra_bad_quote=True))
    result = summarise(session, ids)
    assert result["claims_kept"] == 1 and result["claims_dropped"] == 1 and result["drop_rate"] == 0.5
    assert result["claims_with_unverified_quote"] == 0
    assert hash_files(fixture_papers) == before


def test_user_papers_sample_name_must_exist(fixture_papers, session):
    import pytest

    from claimgraph.eval.user_papers_eval import pick_sample
    from claimgraph.ingest.library import scan_library

    scan_library(session, fixture_papers)
    with pytest.raises(ValueError):
        pick_sample(session, ["no such paper"])


def test_report_section_check_needs_both_citations_under_the_right_heading():
    from claimgraph.eval.fixture_eval import relations_in_right_section

    relations = [{"a": "A", "b": "B", "expected": "contradicts"}, {"a": "A", "b": "C", "expected": "different_setting"}]
    planted = {"A": "x_c1", "B": "y_c1", "C": None}
    right = "## Where the papers conflict\nThey disagree [x_c1] [y_c1]."
    assert relations_in_right_section(right, relations, planted)["relations_in_right_section"] == 1
    wrong = "## Where the papers agree\nThey agree [x_c1] [y_c1]."
    assert relations_in_right_section(wrong, relations, planted)["relations_in_right_section"] == 0
    one = "## Where the papers conflict\nOnly one [x_c1]."
    assert relations_in_right_section(one, relations, planted)["relations_in_right_section"] == 0
