"""Every test gets its own temporary data folder, papers folder and SQLite database.

Nothing here can reach the real MySQL database, the real vector index or the user's papers.
Tests needing MySQL use only the claimgraph_test database.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
FIXTURES = ROOT / "tests" / "fixtures"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["DATABASE_URL"] = "sqlite:///:memory:"  # replaced per test; a safe value at import time


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    from claimgraph.config import reset_settings
    from claimgraph.db import reset_engine
    from claimgraph.embed import embeddings, vectorstore

    papers = tmp_path / "papers"
    papers.mkdir()
    monkeypatch.setenv("CLAIMGRAPH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLAIMGRAPH_PAPERS_DIR", str(papers))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{(tmp_path / 'test.db').as_posix()}")
    vectorstore.close_client()
    reset_settings()
    reset_engine()

    from claimgraph.config import get_settings
    from claimgraph.db import init_db

    get_settings().ensure_dirs()
    init_db()
    yield tmp_path
    embeddings.set_embeddings(None)
    vectorstore.close_client()
    reset_engine()
    reset_settings()


@pytest.fixture
def papers_dir(isolated):
    return isolated / "papers"


@pytest.fixture
def expected() -> dict:
    return json.loads((FIXTURES / "expected.json").read_text(encoding="utf-8"))


@pytest.fixture
def fixture_papers(papers_dir, expected):
    """Copies of the four planted PDFs in the test's papers folder."""
    for paper in expected["papers"]:
        shutil.copy(FIXTURES / paper["file"], papers_dir / paper["file"])
    return papers_dir


@pytest.fixture
def session():
    from claimgraph.db import new_session

    db = new_session()
    yield db
    db.close()


@pytest.fixture
def fake_models(monkeypatch):
    """Replace every model with a fast stand-in, for tests of the plumbing around them."""
    from fakes import FakeEmbeddings, FakeNLI

    from claimgraph import pipeline
    from claimgraph.embed import embeddings

    embeddings.set_embeddings(FakeEmbeddings())
    monkeypatch.setattr(pipeline, "get_nli_model", lambda: FakeNLI())
    monkeypatch.setattr(pipeline, "release_nli_model", lambda: None)
    monkeypatch.setattr(pipeline, "release_embeddings", lambda: None)
    monkeypatch.setattr(pipeline, "unload_llm", lambda: None)
    return True
