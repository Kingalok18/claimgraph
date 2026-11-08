"""Optional online paper search. Refused unless online_search.enabled is true in configs/pipeline.yaml."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import get_settings
from ..db import session_scope
from ..search.search_agent import add_online_papers, run_search_agent
from . import jobs
from .schemas import OnlineSearchRequest

router = APIRouter()


@router.post("/search/online")
def online_search(request: OnlineSearchRequest) -> dict:
    if not get_settings().pipeline["online_search"]["enabled"]:
        raise HTTPException(status_code=409, detail="Online search is switched off. Enable it in configs/pipeline.yaml.")

    def work(context: jobs.JobContext) -> dict:
        context.progress("search", "Searching OpenAlex and arXiv", 0, 0, request.query)
        records = run_search_agent(request.query)
        with session_scope() as session:
            added = add_online_papers(session, records)
        return {"found": len(records), "added": added}

    return {"job_id": jobs.submit("online_search", work)}
