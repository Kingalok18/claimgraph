"""Import PDFs from the papers folder. Files are only ever read; a paper whose file is gone leaves the library."""
from __future__ import annotations

import hashlib
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..embed.vectorstore import remove_claims
from ..models_db import ClaimPair, ClaimTopic, Paper, Report, Section, Topic
from .sections import ParsedPaper

log = logging.getLogger(__name__)
Progress = Callable[[int, int, str], None]


@dataclass
class ScanResult:
    found: int = 0
    new: int = 0
    failed: int = 0
    removed: int = 0
    new_ids: list[str] = field(default_factory=list)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paper_id_for(file_hash: str) -> str:
    return f"p{file_hash[:10]}"


def find_pdfs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def parse_file(path: Path) -> ParsedPaper:
    settings = get_settings()
    if settings.pipeline["parsing"]["backend"] == "grobid":
        from .grobid_loader import parse_pdf

        return parse_pdf(path, settings.grobid_url)
    from .pymupdf_loader import parse_pdf

    return parse_pdf(path)


def _store_parsed(session: Session, paper: Paper, parsed: ParsedPaper) -> None:
    paper.title = (parsed.title or Path(paper.file_path or "").stem)[:2000]
    paper.authors = parsed.authors
    paper.year = parsed.year
    paper.venue = (parsed.venue or "")[:512] or None
    paper.doi = (parsed.doi or "")[:255] or None
    paper.abstract = parsed.abstract
    paper.n_pages = parsed.n_pages
    paper.page_offsets = parsed.page_offsets
    paper.parse_status = "parsed"
    paper.parse_error = None
    session.flush()
    for section in parsed.sections:
        session.add(
            Section(
                paper_id=paper.id,
                name=section.name[:64],
                heading=(section.heading or "")[:512] or None,
                text=section.text,
                char_start=section.char_start,
                page_start=section.page_start,
            )
        )


def scan_library(session: Session, folder: Path | None = None, progress: Progress | None = None) -> ScanResult:
    """Import new PDFs. The same file (by hash) is never imported twice, wherever it sits."""
    folder = Path(folder) if folder else get_settings().papers_dir
    files = find_pdfs(folder)
    result = ScanResult(found=len(files))
    present_hashes: set[str] = set()

    for index, path in enumerate(files, start=1):
        if progress:
            progress(index, len(files), path.name)
        file_hash = file_sha256(path)
        if file_hash in present_hashes:
            continue
        present_hashes.add(file_hash)

        existing = session.scalar(select(Paper).where(Paper.file_hash == file_hash))
        if existing is not None:
            existing.file_path = str(path)
            existing.missing = False
            session.commit()
            continue

        paper = Paper(
            id=paper_id_for(file_hash), file_path=str(path), file_hash=file_hash, source="user", title=path.stem[:2000]
        )
        session.add(paper)
        try:
            _store_parsed(session, paper, parse_file(path))
            result.new += 1
            result.new_ids.append(paper.id)
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the import
            log.warning("Could not parse %s: %s", path.name, exc)
            session.rollback()
            session.add(
                Paper(
                    id=paper_id_for(file_hash),
                    file_path=str(path),
                    file_hash=file_hash,
                    source="user",
                    title=path.stem[:2000],
                    parse_status="failed",
                    parse_error=str(exc)[:1000],
                    extract_status="skipped",
                )
            )
            result.failed += 1
        session.commit()

    # A paper whose file has gone is removed from the library, with its claims. The PDFs themselves are never touched.
    # If the folder itself is gone (a drive unplugged, a wrong setting), nothing is removed.
    if folder.exists():
        rows = session.scalars(select(Paper).where(Paper.source == "user")).all()
        for paper in rows:
            if paper.file_hash not in present_hashes and not (paper.file_path and Path(paper.file_path).exists()):
                remove_paper(session, paper)
                result.removed += 1
        session.commit()
    return result


def remove_paper(session: Session, paper: Paper) -> None:
    """Delete a paper and everything built on its claims: pairs, topic links, stale reports and index points.

    Labels are kept: they are the user's work and store their own copy of what was predicted.
    A topic is regrouped on the next analysis; one left with no claims is deleted now.
    """
    claim_ids = [claim.claim_id for claim in paper.claims]
    if claim_ids:
        touches = or_(ClaimPair.claim_a.in_(claim_ids), ClaimPair.claim_b.in_(claim_ids))
        topic_ids = set(session.scalars(select(ClaimTopic.topic_id).where(ClaimTopic.claim_id.in_(claim_ids))))
        topic_ids |= set(session.scalars(select(ClaimPair.topic_id).where(touches)))
        session.execute(delete(ClaimPair).where(touches))
        session.execute(delete(ClaimTopic).where(ClaimTopic.claim_id.in_(claim_ids)))
        if topic_ids:
            session.execute(delete(Report).where(Report.topic_id.in_(topic_ids)))
            for topic_id in topic_ids:
                left = session.scalar(select(func.count()).select_from(ClaimTopic).where(ClaimTopic.topic_id == topic_id))
                topic = session.get(Topic, topic_id)
                if topic is not None and not left and topic.source == "auto":
                    session.delete(topic)
        remove_claims(claim_ids)
    session.delete(paper)
    session.flush()


_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def save_upload(source, filename: str) -> Path:
    """Copy an uploaded PDF into the uploads folder under a safe, unused name."""
    uploads = get_settings().uploads_dir
    uploads.mkdir(parents=True, exist_ok=True)
    name = _UNSAFE.sub("_", Path(filename).name).strip(" .") or "upload.pdf"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    target = uploads / name
    counter = 1
    while target.exists():
        target = uploads / f"{Path(name).stem}_{counter}.pdf"
        counter += 1
    with target.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return target
