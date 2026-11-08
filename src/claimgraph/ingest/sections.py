"""Shared parsed-paper format, plus heading recognition used by the PyMuPDF backend."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CANONICAL_ORDER = [
    ("abstract", r"^(abstract|summary)$"),
    ("keywords", r"^(key\s?words?|index terms)\b"),
    ("references", r"^(references?|bibliography|works cited)$"),
    ("acknowledgements", r"acknowledge?ments?"),
    ("appendix", r"^(appendix|appendices|supplementa)"),
    ("conclusion", r"conclu|future work|closing remarks"),
    ("discussion", r"^discussions?\b|limitations?|threats to validity"),
    ("results", r"results?\b|experiment|evaluation|performance|comparison|findings|benchmark|empirical"),
    ("related_work", r"related (work|stud|research|literature)|prior (work|art)|literature (review|survey)|state of the art"),
    ("introduction", r"introduction|background|motivation|overview"),
    ("method", r"method|approach|proposed|model|architecture|framework|algorithm|materials|network|design|preliminar|problem|implementation|dataset|data\b"),
]
_COMPILED = [(name, re.compile(pattern, re.I)) for name, pattern in CANONICAL_ORDER]
_NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2})*)\.?\s+(?P<title>[A-Z][^\n]*)$")
_ROMAN = re.compile(r"^(?P<num>[IVX]{1,5})\.?\s+(?P<title>[A-Z][^\n]*)$")
_LETTER = re.compile(r"^(?P<num>[A-H])[.)]\s+(?P<title>[A-Z][^\n]*)$")
_INLINE_ABSTRACT = re.compile(r"^(abstract|summary)\s*[—–:.\-]+\s*(?=\S)", re.I)
_NOT_HEADING_START = re.compile(r"^(fig(ure)?|table|algorithm|eq(uation)?|http)\b", re.I)


@dataclass
class ParsedSection:
    name: str  # canonical: abstract, results, ...
    heading: str | None
    text: str
    char_start: int
    page_start: int | None


@dataclass
class ParsedPaper:
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    abstract: str | None = None
    n_pages: int = 0
    sections: list[ParsedSection] = field(default_factory=list)
    page_offsets: list[int] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "".join(section.text for section in self.sections)


@dataclass
class Heading:
    title: str
    canonical: str | None  # None when the title matches no known kind of section
    level: int  # 1 = top level, 2 = subsection
    inline_rest: str = ""  # body text on the same line, e.g. "Abstract—We study ..."


def canonical_name(title: str) -> str | None:
    clean = re.sub(r"[^A-Za-z ]+", " ", title).strip()
    for name, pattern in _COMPILED:
        if pattern.search(clean):
            return name
    return None


def classify_heading(line: str, size: float, bold: bool, body_size: float) -> Heading | None:
    """Decide whether a line is a section heading, from its numbering, wording and font."""
    text = line.strip()
    if not text:
        return None

    inline = _INLINE_ABSTRACT.match(text)
    if inline:
        return Heading(title=inline.group(1).title(), canonical="abstract", level=1, inline_rest=text[inline.end() :])

    if len(text) > 90 or len(text.split()) > 12 or _NOT_HEADING_START.match(text):
        return None
    if text[-1] in ",;:":
        return None
    prominent = size >= body_size * 1.05 or bold

    for pattern, level_of in ((_NUMBERED, None), (_ROMAN, 1), (_LETTER, 2)):
        match = pattern.match(text)
        if not match:
            continue
        title = match.group("title").strip()
        if title.endswith(".") or not prominent:
            continue
        if not re.search(r"[A-Za-z]{3}", title):
            continue
        level = level_of if level_of is not None else match.group("num").count(".") + 1
        return Heading(title=title, canonical=canonical_name(title), level=level)

    bare = text.rstrip(".:").strip()
    known = canonical_name(bare)
    # An unnumbered heading has to be short. Long sentences that merely contain "results" are body text.
    if known and len(bare.split()) <= 6 and (prominent or bare.isupper()):
        return Heading(title=bare.title() if bare.isupper() else bare, canonical=known, level=1)
    if size >= body_size * 1.15 and not text.endswith(".") and text[0].isupper() and len(text.split()) <= 10:
        return Heading(title=bare, canonical=None, level=1)
    return None


def parent_aware_name(heading: Heading, current_top: str) -> str:
    """Subsections inherit their parent ('4.2 Metrics' under Experiments is still results)."""
    if heading.level >= 2:
        return current_top
    return heading.canonical or "other"
