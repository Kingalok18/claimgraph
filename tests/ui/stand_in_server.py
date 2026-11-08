"""The real app and frontend, with stand-in models so that the browser test takes minutes, not hours.

Runs on a separate port with temporary storage. The real-model path is covered by
tests/test_pipeline_fixtures.py and by the run on the user's papers.

Importing this file replaces the real models, so only ever start it as its own process. Its name must
not match test_*.py or *_test.py: pytest would import it while collecting and every test would get stand-ins.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

import uvicorn  # noqa: E402
from fakes import FakeEmbeddings, FakeNLI, fake_extract_chain  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402

from claimgraph import pipeline  # noqa: E402
from claimgraph.api import main, routes_library  # noqa: E402
from claimgraph.embed import embeddings  # noqa: E402
from claimgraph.report import report_chain  # noqa: E402

READY = {"ok": True, "fatal": False, "provider": "hf_gguf", "model": "stand-in model for the browser test", "detail": "ready"}

embeddings.set_embeddings(FakeEmbeddings())
pipeline.get_nli_model = lambda: FakeNLI()
pipeline.release_nli_model = lambda: None
pipeline.release_embeddings = lambda: None
pipeline.unload_llm = lambda: None
pipeline.build_extract_chain = fake_extract_chain
pipeline.build_label_chain = lambda: RunnableLambda(lambda _: "Dropout and test accuracy")
report_chain.build_report_chain = lambda: RunnableLambda(lambda _: "No citations, so the app builds the map from stored relations.")
routes_library.check_llm = lambda: READY
main.check_llm = lambda: READY
main._embedding_status = lambda: {
    "embeddings": {"model": "BAAI/bge-m3", "ok": True},
    "nli": {"model": "cross-encoder/nli-deberta-v3-base", "ok": True, "finetuned": False},
}

if __name__ == "__main__":
    uvicorn.run(main.create_app(), host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
