"""PDF parsing through GROBID, using LangChain's GenericLoader and GrobidParser.

Needs the GROBID service (docker-compose). This install uses PyMuPDF; set parsing.backend to
grobid in configs/pipeline.yaml to switch. The output format matches pymupdf_loader.
"""
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path

import httpx

from .pymupdf_loader import parse_pdf as parse_with_pymupdf
from .sections import ParsedPaper, ParsedSection, canonical_name

log = logging.getLogger(__name__)


def grobid_available(url: str, timeout: float = 2.0) -> bool:
    try:
        return httpx.get(f"{url.rstrip('/')}/api/isalive", timeout=timeout).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _first_page(value) -> int | None:
    """GrobidParser stores pages as a tuple, or its text form such as "(3, 4)"."""
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        return int(parsed[0]) if isinstance(parsed, (tuple, list)) else int(parsed)
    except (ValueError, SyntaxError, TypeError, IndexError):
        found = re.search(r"\d+", str(value))
        return int(found.group(0)) if found else None


def documents_to_parsed(documents: list, fallback: ParsedPaper | None = None) -> ParsedPaper:
    """Group GROBID's segments by section title and build the shared format."""
    grouped: list[tuple[str, int | None, list[str]]] = []
    title = None
    for document in documents:
        meta = document.metadata
        title = title or meta.get("paper_title")
        section_title = (meta.get("section_title") or "Body").strip()
        page = _first_page(meta.get("pages"))
        if grouped and grouped[-1][0] == section_title:
            grouped[-1][2].append(document.page_content.strip())
        else:
            grouped.append((section_title, page, [document.page_content.strip()]))

    sections: list[ParsedSection] = []
    offset = 0
    current_top = "other"
    if fallback and fallback.abstract:
        text = f"Abstract\n{fallback.abstract}\n\n"
        sections.append(ParsedSection("abstract", "Abstract", text, 0, 1))
        offset = len(text)
    for section_title, page, parts in grouped:
        known = canonical_name(section_title)
        current_top = known or current_top
        text = f"{section_title}\n{' '.join(part for part in parts if part)}\n\n"
        sections.append(ParsedSection(current_top, section_title, text, offset, page))
        offset += len(text)

    # GROBID segments give no character positions per page, so offsets come from section starts.
    page_offsets: list[int] = []
    for section in sections:
        page = section.page_start or 1
        while len(page_offsets) < page:
            page_offsets.append(section.char_start)

    base = fallback or ParsedPaper()
    return ParsedPaper(
        title=title or base.title,
        authors=base.authors,
        year=base.year,
        venue=base.venue,
        doi=base.doi,
        abstract=base.abstract,
        n_pages=base.n_pages,
        sections=sections,
        page_offsets=page_offsets,
    )


def parse_pdf(path: str | Path, grobid_url: str) -> ParsedPaper:
    from langchain_community.document_loaders.generic import GenericLoader
    from langchain_community.document_loaders.parsers import GrobidParser

    path = Path(path)
    loader = GenericLoader.from_filesystem(
        str(path.parent),
        glob=path.name,
        suffixes=[path.suffix],
        parser=GrobidParser(
            segment_sentences=True,
            grobid_server=f"{grobid_url.rstrip('/')}/api/processFulltextDocument",
        ),
    )
    documents = loader.load()
    if not documents:
        raise ValueError("GROBID returned no text for this PDF.")
    # Header details and the abstract come from PyMuPDF, because GrobidParser returns body text only.
    return documents_to_parsed(documents, fallback=parse_with_pymupdf(path))
