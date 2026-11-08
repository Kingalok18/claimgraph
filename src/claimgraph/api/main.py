"""FastAPI application. Serves the API and the built frontend from one local address."""
from __future__ import annotations

import logging
import mimetypes
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..config import ROOT, get_settings
from ..db import check_database, init_db
from ..llm import check_llm
from . import jobs, routes_labels, routes_library, routes_pairs, routes_report, routes_topics

log = logging.getLogger(__name__)
FRONTEND_DIST = ROOT / "frontend" / "dist"
# The Windows registry can map these to text/plain, and browsers then refuse to run the scripts.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")


def _embedding_status() -> dict:
    """Is the embedding model cached? Checked without loading it."""
    config = get_settings().models["embeddings"]
    nli = get_settings().models["nli"]

    def cached(repo: str) -> bool:
        try:
            from huggingface_hub import try_to_load_from_cache

            return isinstance(try_to_load_from_cache(repo, "config.json"), str)
        except Exception:  # noqa: BLE001
            return False

    return {
        "embeddings": {"model": config["model_id"], "ok": cached(config["model_id"])},
        "nli": {
            "model": nli["base_model_id"],
            "ok": get_settings().nli_finetuned_path.exists() or cached(nli["base_model_id"]),
            "finetuned": get_settings().nli_finetuned_path.exists(),
        },
    }


def health_payload() -> dict:
    settings = get_settings()
    database = check_database()
    llm = check_llm()
    models = _embedding_status()
    problems: list[dict] = []
    if not database["ok"]:
        problems.append({"service": "database", "blocking": True, "detail": database["detail"]})
    if not llm["ok"]:
        problems.append({"service": "language model", "blocking": bool(llm.get("fatal")), "detail": llm["detail"]})
    if not models["embeddings"]["ok"]:
        problems.append(
            {
                "service": "embedding model",
                "blocking": False,
                "detail": f"{models['embeddings']['model']} is not downloaded yet. It downloads on first use, or run: python scripts/run.py setup",
            }
        )
    if not models["nli"]["ok"]:
        problems.append(
            {
                "service": "comparison model",
                "blocking": False,
                "detail": f"{models['nli']['model']} is not downloaded yet. It downloads on first use, or run: python scripts/run.py setup",
            }
        )
    return {
        "status": "ok" if not any(problem["blocking"] for problem in problems) else "setup_needed",
        "app_name": settings.app_name,
        "default_theme": settings.pipeline.get("default_theme", "light"),
        "services": {
            "database": database,
            "llm": llm,
            "embeddings": models["embeddings"],
            "nli": models["nli"],
            "vector_store": {"ok": True, "mode": "server" if settings.qdrant_url else "local", "path": str(settings.qdrant_path)},
        },
        "setup_problems": problems,
        "settings": {
            "papers_dir": str(settings.papers_dir),
            "parsing_backend": settings.pipeline["parsing"]["backend"],
            "online_search": bool(settings.pipeline["online_search"]["enabled"]),
        },
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    settings.ensure_dirs()
    try:
        init_db()
        interrupted = jobs.mark_interrupted()
        if interrupted:
            log.info("Marked %d unfinished jobs as interrupted", interrupted)
        if settings.pipeline.get("scan_on_start", True):
            jobs.submit("scan", routes_library.scan_job)
    except Exception as exc:  # noqa: BLE001 - /health and the setup screen explain the problem
        log.warning("Database is not ready: %s", exc)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="ClaimGraph", lifespan=lifespan)
    # Only the Vite dev server needs this. The built app is served from the same address.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return health_payload()

    app.include_router(routes_library.router)
    app.include_router(routes_topics.router)
    app.include_router(routes_pairs.router)
    app.include_router(routes_report.router)
    app.include_router(routes_labels.router)

    from . import routes_online

    app.include_router(routes_online.router)

    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        page = Path(FRONTEND_DIST / "index.html")
        if not page.exists():
            raise HTTPException(status_code=404, detail="The frontend is not built. Run: python scripts/run.py setup")
        return FileResponse(page)

    return app


app = create_app()
