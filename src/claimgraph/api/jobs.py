"""Background jobs. One worker thread: the models must not compete for this machine's memory."""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from sqlalchemy import select

from ..db import session_scope
from ..models_db import Job, utcnow
from ..pipeline import Cancelled

log = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="claimgraph-job")
_cancel_flags: dict[int, threading.Event] = {}
ACTIVE = ("queued", "running")


class JobContext:
    def __init__(self, job_id: int, flag: threading.Event) -> None:
        self.job_id = job_id
        self._flag = flag

    def should_cancel(self) -> bool:
        return self._flag.is_set()

    def progress(self, step: str, label: str, done: int = 0, total: int = 0, current: str | None = None) -> None:
        with session_scope() as session:
            job = session.get(Job, self.job_id)
            if job is not None:
                job.progress = {"step": step, "label": label, "done": done, "total": total, "current": current}


def active_job(kind: str | None = None) -> Job | None:
    with session_scope() as session:
        query = select(Job).where(Job.status.in_(ACTIVE)).order_by(Job.id)
        if kind:
            query = query.where(Job.kind == kind)
        job = session.scalar(query)
        if job is not None:
            session.expunge(job)
        return job


def submit(kind: str, work: Callable[[JobContext], dict | None]) -> int:
    """Queue work. If a job of this kind is already waiting or running, that job is returned instead."""
    existing = active_job(kind)
    if existing is not None:
        return existing.id
    with session_scope() as session:
        job = Job(kind=kind, status="queued", progress={"step": "queued", "label": "Waiting to start", "done": 0, "total": 0})
        session.add(job)
        session.flush()
        job_id = job.id
    flag = threading.Event()
    _cancel_flags[job_id] = flag
    _executor.submit(_run, job_id, work, flag)
    return job_id


def _finish(job_id: int, status: str, summary: dict | None = None, error: str | None = None) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = status
        job.error = error
        job.finished_at = utcnow()
        job.progress = {**(job.progress or {}), "summary": summary}


def _run(job_id: int, work, flag: threading.Event) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.status = "running"
        job.started_at = utcnow()
    try:
        summary = work(JobContext(job_id, flag))
        _finish(job_id, "done", summary=summary)
    except Cancelled:
        _finish(job_id, "cancelled")
    except Exception as exc:  # noqa: BLE001 - the error is shown to the user
        log.exception("Job %s failed", job_id)
        _finish(job_id, "failed", error=str(exc)[:1000])
    finally:
        _cancel_flags.pop(job_id, None)


def cancel(job_id: int) -> bool:
    flag = _cancel_flags.get(job_id)
    if flag is None:
        return False
    flag.set()
    return True


def mark_interrupted() -> int:
    """Jobs left running by a previous start can never finish. Mark them, so the UI does not wait."""
    with session_scope() as session:
        jobs = session.scalars(select(Job).where(Job.status.in_(ACTIVE))).all()
        for job in jobs:
            job.status = "interrupted"
            job.finished_at = utcnow()
        return len(jobs)
