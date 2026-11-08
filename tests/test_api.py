"""Every endpoint in spec Section 8.9, against stand-in models."""
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from conftest import FIXTURES
from fakes import fake_extract_chain
from claimgraph.config import get_settings
from claimgraph.db import session_scope
from claimgraph.models_db import ClaimPair, Label, Topic


@pytest.fixture
def client(monkeypatch):
    from claimgraph.api import main

    monkeypatch.setitem(get_settings().pipeline, "scan_on_start", False)
    with TestClient(main.create_app()) as test_client:
        yield test_client


def wait(client, job_id, seconds=30):
    deadline = time.time() + seconds
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {job}")


@pytest.fixture
def analysed(client, fixture_papers, fake_models, monkeypatch):
    """Library analysed end to end through the API."""
    from langchain_core.runnables import RunnableLambda

    from claimgraph import pipeline
    from claimgraph.api import routes_library

    monkeypatch.setattr(pipeline, "build_extract_chain", fake_extract_chain)
    monkeypatch.setattr(pipeline, "build_label_chain", lambda: RunnableLambda(lambda _: "Dropout and accuracy"))
    monkeypatch.setattr(routes_library, "check_llm", lambda: {"ok": True, "fatal": False})
    assert client.post("/library/scan").json()["new"] == 4
    job = wait(client, client.post("/analyze").json()["job_id"])
    assert job["status"] == "done", job
    return client


def test_health_reports_each_service(client):
    data = client.get("/health").json()
    assert data["services"]["database"]["ok"] is True
    for service in ("llm", "embeddings", "nli", "vector_store"):
        assert service in data["services"]
    assert data["app_name"] == "ClaimGraph" and data["default_theme"] == "light"


def test_health_explains_a_broken_database(client, monkeypatch):
    from claimgraph.api import main

    monkeypatch.setattr(main, "check_database", lambda: {"ok": False, "kind": "mysql", "detail": "MySQL is not reachable."})
    data = client.get("/health").json()
    assert data["status"] == "setup_needed"
    assert any(problem["service"] == "database" and problem["blocking"] for problem in data["setup_problems"])


def test_scan_and_library(client, fixture_papers):
    assert client.post("/library/scan").json()["new"] == 4
    data = client.get("/library").json()
    assert len(data["papers"]) == 4 and data["pending"] == 4
    assert client.post("/library/scan").json()["new"] == 0


def test_background_scan_reports_progress_wording(client, fixture_papers):
    job = wait(client, client.post("/library/scan?background=true").json()["job_id"])
    assert job["status"] == "done" and job["progress"]["label"] == "Reading papers 4 of 4"
    assert job["progress"]["summary"]["new"] == 4


def test_upload_pdf(client, papers_dir, expected):
    pdf = (FIXTURES / expected["papers"][0]["file"]).read_bytes()
    response = client.post("/library/upload", files={"file": ("my paper.pdf", pdf, "application/pdf")})
    assert response.status_code == 200 and response.json()["new"] == 1
    assert (papers_dir / "uploads" / "my paper.pdf").read_bytes() == pdf


@pytest.mark.parametrize("name,content", [("a.txt", b"%PDF-1"), ("a.pdf", b"hello there")])
def test_upload_rejects_non_pdfs(client, name, content):
    assert client.post("/library/upload", files={"file": (name, content)}).status_code == 400


def test_paper_detail_and_pdf(analysed):
    paper = analysed.get("/library").json()["papers"][0]
    detail = analysed.get(f"/papers/{paper['id']}").json()
    assert detail["claims"] == 1 and detail["claims_list"][0]["quote_verified"] is True
    assert {"Abstract", "Results"} <= {section["heading"] for section in detail["sections"]}
    pdf = analysed.get(f"/papers/{paper['id']}/pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")


def test_unknown_ids_are_404(client):
    for path in ("/papers/nope", "/papers/nope/pdf", "/jobs/999", "/topics/999/graph", "/topics/999/report", "/pairs/999"):
        assert client.get(path).status_code == 404


def test_analyze_refuses_clearly_when_the_model_cannot_run(client, monkeypatch):
    from claimgraph.api import routes_library

    monkeypatch.setattr(routes_library, "check_llm", lambda: {"ok": False, "fatal": True, "detail": "Install it."})
    response = client.post("/analyze")
    assert response.status_code == 503 and response.json()["detail"] == "Install it."


def test_topics_counts_and_graph(analysed):
    topics = analysed.get("/topics").json()["topics"]
    assert len(topics) == 1
    topic = topics[0]
    assert topic["label"] == "Dropout and accuracy" and topic["claims"] == 4
    assert topic["contradicts"] >= 2 and topic["different_setting"] >= 2 and topic["supports"] >= 1
    graph = analysed.get(f"/topics/{topic['id']}/graph").json()
    assert len(graph["elements"]["nodes"]) == 4 and graph["elements"]["edges"]
    assert len(graph["papers"]) == 4


def test_pair_view_and_label_override(analysed):
    with session_scope() as session:
        pair = session.scalar(select(ClaimPair).where(ClaimPair.relation == "contradicts"))
        pair_id, topic_id = pair.id, pair.topic_id
    data = analysed.get(f"/pairs/{pair_id}").json()
    assert data["relation"] == "contradicts" and data["user_label"] is None
    assert data["a"]["quote"] and data["a"]["page"] == 1 and data["b"]["paper_title"]
    assert data["field_status"]["dataset"] == "same"

    corrected = analysed.post(f"/pairs/{pair_id}/label", json={"relation": "different_setting", "note": "checked"}).json()
    assert corrected["user_label"] == "different_setting" and corrected["relation"] == "contradicts"
    edge = next(
        edge["data"] for edge in analysed.get(f"/topics/{topic_id}/graph").json()["elements"]["edges"] if edge["data"]["pair_id"] == pair_id
    )
    assert edge["relation"] == "different_setting"
    labels_file = get_settings().labels_dir / "pair_labels.jsonl"
    assert labels_file.exists() and "different_setting" in labels_file.read_text(encoding="utf-8")
    assert analysed.post(f"/pairs/{pair_id}/label", json={"relation": "wrong"}).status_code == 422


def test_report_citations_all_resolve(analysed, monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from claimgraph.report import report_chain

    topic = analysed.get("/topics").json()["topics"][0]
    with session_scope() as session:
        pair = session.scalar(select(ClaimPair).where(ClaimPair.relation == "contradicts"))
        a, b = pair.claim_a, pair.claim_b
    text = f"## Where the papers conflict\nThey disagree on accuracy [{a}] [{b}]. An invented line [fake_c1]."
    monkeypatch.setattr(report_chain, "build_report_chain", lambda: RunnableLambda(lambda _: text))
    data = analysed.get(f"/topics/{topic['id']}/report").json()
    # The model's backed sentence stays, its invented one goes, and relations it skipped are added from stored data.
    assert "They disagree on accuracy" in data["text"] and "invented" not in data["text"]
    assert {a, b} <= set(data["cited_claim_ids"]) and data["removed_count"] == 1
    assert set(data["claims"]) == set(data["cited_claim_ids"])  # every citation resolves to a stored claim
    assert "fake_c1" not in data["cited_claim_ids"]

    markdown = analysed.get(f"/topics/{topic['id']}/report.md")
    assert markdown.status_code == 200 and "## Sources" in markdown.text and "[1]" in markdown.text and a not in markdown.text
    pdf = analysed.get(f"/topics/{topic['id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF-")


def test_search_claims_and_papers(analysed):
    data = analysed.get("/search?q=dropout").json()
    assert data["claims"] and data["papers"]
    assert analysed.get("/search?q=a").json() == {"claims": [], "papers": []}
    assert analysed.get("/search?q=100%25").json()["claims"] == []  # % is a literal, not a wildcard


def test_question_endpoint(analysed, monkeypatch):
    monkeypatch.setitem(get_settings().pipeline["topics"], "question_similarity", 0.1)
    job = wait(analysed, analysed.post("/topics/question", json={"question": "Does dropout change accuracy?"}).json()["job_id"])
    assert job["status"] == "done", job
    topic_id = job["progress"]["summary"]["topic_id"]
    with session_scope() as session:
        assert session.get(Topic, topic_id).source == "question"
    assert analysed.post("/topics/question", json={"question": "x"}).status_code == 422


def test_label_queue_stats_and_claim_labels(analysed):
    queue = analysed.get("/labels/queue?kind=pair").json()["items"]
    assert queue
    first = analysed.get(f"/pairs/{queue[0]['id']}").json()
    assert first["relation"] == "contradicts"  # conflicts come first
    analysed.post(f"/pairs/{queue[0]['id']}/label", json={"relation": "contradicts"})
    assert queue[0]["id"] not in [item["id"] for item in analysed.get("/labels/queue?kind=pair").json()["items"]]

    claims = analysed.get("/labels/queue?kind=extraction").json()["items"]
    assert analysed.post("/labels/claim", json={"claim_id": claims[0]["claim_id"], "verdict": "correct"}).status_code == 200
    assert analysed.post("/labels/claim", json={"claim_id": "ghost", "verdict": "correct"}).status_code == 404
    stats = analysed.get("/labels/stats").json()
    assert stats["pair_labels"] == 1 and stats["claim_labels"] == 1 and stats["claims_total"] == 4
    with session_scope() as session:
        assert session.query(Label).count() == 2


def test_online_search_is_refused_when_off(client):
    assert client.post("/search/online", json={"query": "image forgery"}).status_code == 409


def test_active_job_endpoint(client):
    assert client.get("/jobs/active").json() == {"job": None}
    assert client.post("/jobs/999/cancel").json() == {"cancelling": False}


def test_interrupted_jobs_are_marked_on_start(client):
    from claimgraph.api import jobs
    from claimgraph.models_db import Job

    with session_scope() as session:
        session.add(Job(kind="analyze", status="running"))
    assert jobs.mark_interrupted() == 1
    assert client.get("/jobs/active").json() == {"job": None}
