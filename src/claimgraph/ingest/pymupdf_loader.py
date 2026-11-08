"""PDF parsing with PyMuPDF: clean text, headings from fonts and numbering, page offsets.

PyMuPDFLoader (LangChain) supplies the document metadata. The body goes through PyMuPDF's
line-level API because the loader does not expose font sizes, which heading detection needs.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import fitz

from .sections import ParsedPaper, ParsedSection, classify_heading, parent_aware_name

log = logging.getLogger(__name__)

_DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.I)
_ARXIV_NAME = re.compile(r"^(\d{2})(\d{2})\.\d{4,5}")
_YEAR = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
_VENUES = re.compile(
    r"\b(CVPR|ICCV|ECCV|NeurIPS|NIPS|ICML|ICLR|AAAI|IJCAI|ACM MM|ACL|EMNLP|TPAMI|TIFS|TIP|ICDAR|IEEE Access|WACV|arXiv)\b"
)
_JUNK_TITLE = re.compile(r"(untitled|\.pdf$|\.dvi$|\.tex$|\.docx?$|microsoft word|^arxiv:|^doi:|^\d+$|^template)", re.I)
_FLAGS = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_IMAGES
MIN_TEXT_CHARS = 200


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float
    page: int
    block: int


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", "").replace("­", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def _page_lines(page: fitz.Page, page_index: int) -> list[Line]:
    lines: list[Line] = []
    for block_index, block in enumerate(page.get_text("dict", flags=_FLAGS).get("blocks", [])):
        for line in block.get("lines", []):
            if line.get("dir", (1, 0))[0] < 0.9:  # rotated text: the arXiv stamp, margin notes
                continue
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            text = _clean("".join(span["text"] for span in line["spans"]))
            if not text:
                continue
            biggest = max(spans, key=lambda span: len(span["text"]))
            bold = bool(biggest.get("flags", 0) & 16) or "bold" in biggest.get("font", "").lower()
            lines.append(
                Line(text, round(biggest.get("size", 0), 1), bold, line["bbox"][1], line["bbox"][3], page_index, block_index)
            )
    return lines


def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", text.lower())


def _at_margin(line: Line, height: float) -> bool:
    return line.y0 < height * 0.09 or line.y1 > height * 0.93


def _remove_running_text(pages: list[list[Line]], heights: list[float]) -> list[list[Line]]:
    """Drop page numbers and headers or footers that repeat across pages."""
    counts: Counter = Counter()
    for index, lines in enumerate(pages):
        counts.update({_signature(line.text) for line in lines if _at_margin(line, heights[index])})
    limit = max(3, int(len(pages) * 0.4))
    repeated = {key for key, count in counts.items() if count >= limit}

    output = []
    for index, lines in enumerate(pages):
        kept = []
        for line in lines:
            if _at_margin(line, heights[index]) and (
                _signature(line.text) in repeated or re.fullmatch(r"[\d\s.\-–]+", line.text)
            ):
                continue
            kept.append(line)
        output.append(kept)
    return output


def _body_size(pages: list[list[Line]]) -> float:
    counts: Counter = Counter()
    for lines in pages:
        for line in lines:
            counts[line.size] += len(line.text)
    return counts.most_common(1)[0][0] if counts else 10.0


def _join_lines(parts: list[str]) -> str:
    """Rejoin wrapped lines, undoing hyphenation only when the next line continues a word."""
    text = ""
    for part in parts:
        if text.endswith("-") and part[:1].islower():
            text = text[:-1] + part
        elif text:
            text += " " + part
        else:
            text = part
    return text


def _metadata(path: Path) -> dict:
    try:
        from langchain_community.document_loaders import PyMuPDFLoader

        first = next(iter(PyMuPDFLoader(str(path)).lazy_load()), None)
        return dict(first.metadata) if first else {}
    except Exception:  # noqa: BLE001 - metadata is a nicety; the file's own is the fallback
        log.debug("PyMuPDFLoader metadata failed for %s", path, exc_info=True)
        return {}


def _good_title(value: str | None) -> str | None:
    if not value:
        return None
    value = _clean(value)
    if len(value) < 12 or _JUNK_TITLE.search(value):
        return None
    return value


def _title_from_first_page(lines: list[Line], height: float) -> tuple[str | None, int]:
    """The title is the largest text in the top part of page one. Returns text and its last line index."""
    candidates = [line for line in lines if line.y0 < height * 0.45 and len(line.text) > 3]
    if not candidates:
        return None, -1
    largest = max(line.size for line in candidates)
    picked: list[str] = []
    last = -1
    for index, line in enumerate(lines):
        if line in candidates and line.size >= largest - 0.5:
            if picked and index - last > 2:
                break
            picked.append(line.text)
            last = index
    return (_join_lines(picked)[:400] or None), last


def _authors_from_first_page(lines: list[Line], title_end: int) -> str | None:
    gathered = []
    for line in lines[title_end + 1 : title_end + 6]:
        if re.match(r"^(abstract|summary|introduction|\d+\.?\s)", line.text, re.I):
            break
        if "@" in line.text or re.search(r"\b(university|institute|department|school|laboratory|college)\b", line.text, re.I):
            break
        gathered.append(line.text)
    joined = " ".join(gathered).strip()
    return joined[:400] if joined else None


def _year(path: Path, first_page_text: str, metadata: dict) -> int | None:
    match = _ARXIV_NAME.match(path.stem)
    if match and 1 <= int(match.group(2)) <= 12:
        return 2000 + int(match.group(1))
    years = Counter(int(y) for y in _YEAR.findall(first_page_text))
    if years:
        return years.most_common(1)[0][0]
    raw = str(metadata.get("creationdate") or metadata.get("creationDate") or "")
    found = _YEAR.search(raw)
    return int(found.group(1)) if found else None


def parse_pdf(path: str | Path) -> ParsedPaper:
    path = Path(path)
    metadata = _metadata(path)
    with fitz.open(path) as document:
        if not metadata:
            metadata = {key.lower(): value for key, value in (document.metadata or {}).items()}
        n_pages = document.page_count
        heights = [page.rect.height for page in document]
        pages = [_page_lines(page, index) for index, page in enumerate(document)]

    pages = _remove_running_text(pages, heights)
    body_size = _body_size(pages)
    first_lines = pages[0] if pages else []
    first_text = " ".join(line.text for line in first_lines)
    if sum(len(line.text) for lines in pages for line in lines) < MIN_TEXT_CHARS:
        raise ValueError("No usable text layer. This PDF is probably a scan and needs OCR first.")

    title, title_end = _title_from_first_page(first_lines, heights[0]) if first_lines else (None, -1)
    title_lines = first_lines[: title_end + 1] if title_end >= 0 else []
    title = _good_title(metadata.get("title")) or title or path.stem

    # Assemble the full text, recording where each page and each heading begins.
    pieces: list[str] = []
    page_offsets: list[int] = []
    boundaries: list[tuple[int, str, str | None, int]] = [(0, "front", None, 1)]
    current_top = "front"
    in_references = False
    offset = 0

    for page_index, lines in enumerate(pages):
        page_offsets.append(offset)
        buffer: list[str] = []
        block = None
        for line in lines:
            if page_index == 0 and line in title_lines:
                piece = line.text + "\n"
                pieces.append(piece)
                offset += len(piece)
                continue
            if line.block != block:
                if buffer:
                    piece = _join_lines(buffer) + "\n\n"
                    pieces.append(piece)
                    offset += len(piece)
                    buffer = []
                block = line.block
            heading = None if in_references else classify_heading(line.text, line.size, line.bold, body_size)
            if heading is None:
                buffer.append(line.text)
                continue
            if buffer:
                piece = _join_lines(buffer) + "\n\n"
                pieces.append(piece)
                offset += len(piece)
                buffer = []
            name = parent_aware_name(heading, current_top)
            if heading.level == 1:
                current_top = name
            boundaries.append((offset, name, heading.title, page_index + 1))
            in_references = name == "references"
            if heading.inline_rest:
                piece = heading.title + "\n"
                buffer.append(heading.inline_rest)
            else:
                piece = line.text + "\n"
            pieces.append(piece)
            offset += len(piece)
        if buffer:
            piece = _join_lines(buffer) + "\n\n"
            pieces.append(piece)
            offset += len(piece)

    full_text = "".join(pieces)
    sections: list[ParsedSection] = []
    for index, (start, name, heading, page) in enumerate(boundaries):
        end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(full_text)
        if end > start:
            sections.append(ParsedSection(name, heading, full_text[start:end], start, page))

    abstract = next((section.text for section in sections if section.name == "abstract"), None)
    if abstract:
        abstract = re.sub(r"^\s*(abstract|summary)\s*[—–:.\-]*\s*", "", abstract, flags=re.I).strip()

    doi = _DOI.search(first_text + " " + str(metadata.get("doi") or ""))
    venue = _VENUES.search(first_text)
    author = metadata.get("author")
    authors = _clean(author) if author and len(_clean(author)) > 3 else None
    if not authors and title_end >= 0:
        authors = _authors_from_first_page(first_lines, title_end)

    return ParsedPaper(
        title=title,
        authors=authors,
        year=_year(path, first_text, metadata),
        venue=venue.group(1) if venue else ("arXiv" if _ARXIV_NAME.match(path.stem) else None),
        doi=doi.group(0).rstrip(".,;)") if doi else None,
        abstract=abstract,
        n_pages=n_pages,
        sections=sections,
        page_offsets=page_offsets,
    )
