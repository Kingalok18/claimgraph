import pytest
from sqlalchemy import inspect, text

from claimgraph.config import ROOT, get_settings, load_settings, reset_settings
from claimgraph.db import check_database, get_engine, make_engine, reset_engine
from claimgraph.models_db import Base


def test_configs_load_with_required_keys():
    settings = load_settings(ROOT)
    assert settings.models["llm"]["provider"]
    assert settings.models["embeddings"]["model_id"] == "BAAI/bge-m3"
    assert settings.pipeline["pairs"]["neighbours_per_claim"] == 10
    assert settings.pipeline["quote_verify"]["fuzzy_threshold"] == 95


def test_user_choices_are_applied():
    settings = load_settings(ROOT)
    assert settings.pipeline["parsing"]["backend"] == "pymupdf"
    assert settings.pipeline["online_search"]["enabled"] is False
    assert settings.models["llm"]["provider"] == "hf_gguf"


def test_env_overrides_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAIMGRAPH_PAPERS_DIR", str(tmp_path))
    reset_settings()
    assert get_settings().papers_dir == tmp_path
    assert get_settings().uploads_dir == tmp_path / "uploads"


def test_all_spec_tables_exist():
    tables = set(inspect(get_engine()).get_table_names())
    assert {"papers", "sections", "claims", "topics", "claim_topics", "claim_pairs", "reports", "jobs", "labels", "llm_calls"} <= tables


def test_database_check_reports_ok():
    assert check_database() == {"ok": True, "kind": "sqlite", "detail": "connected"}


def test_database_check_reports_problem_without_leaking_password(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://someone:s3cretpw@127.0.0.1:1/none?connect_timeout=1")
    reset_settings()
    reset_engine()
    status = check_database()
    assert status["ok"] is False and status["kind"] == "mysql"
    assert "s3cretpw" not in status["detail"]


@pytest.mark.mysql
def test_schema_works_on_mysql():
    url = load_settings(ROOT).test_mysql_url
    if not url:
        pytest.skip("TEST_MYSQL_URL is not set")
    from sqlalchemy.orm import Session

    from claimgraph.models_db import ClaimRow, Paper, Section

    engine = make_engine(url)
    try:
        assert engine.url.database == "claimgraph_test"  # never the real database
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        long_text = "word " * 40000  # beyond MySQL TEXT's 64 KB, as long sections can be
        with Session(engine) as session:
            session.add(Paper(id="p1", file_hash="h" * 64, title="t", page_offsets=[0, 10]))
            session.flush()
            session.add(Section(paper_id="p1", name="results", text=long_text, char_start=0))
            session.add(
                ClaimRow(
                    claim_id="p1_c1", paper_id="p1", text="Déjà vu 𝛼 test", intervention="i", outcome="o",
                    direction="increase", conditions={"dataset": "α-set"}, hedged=False, section="results",
                    quote="q", quote_verified=True,
                )
            )
            session.commit()
        with Session(engine) as session:
            assert len(session.scalar(text("select text from sections"))) == len(long_text)
            row = session.get(ClaimRow, "p1_c1")
            assert row.text == "Déjà vu 𝛼 test"  # four-byte characters survive utf8mb4
            assert row.conditions == {"dataset": "α-set"}
            assert session.get(Paper, "p1").page_offsets == [0, 10]
        Base.metadata.drop_all(engine)
    finally:
        engine.dispose()
