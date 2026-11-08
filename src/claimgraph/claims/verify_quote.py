"""Check that a quote from the LLM really exists in the paper, and find where.

Exact search first (ignoring whitespace and case), then fuzzy search over sentence windows.
On a match the quote is replaced by the paper's own text, so what we store is what the paper says.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

from rapidfuzz import fuzz

# The parser joins wrapped lines, so a line break left in the text is a real boundary: a heading or a new block.
_SENTENCE_END = re.compile(r"\s*\n\s*|(?<=[.!?])\s+(?=[A-Z0-9(\[\"'])")
_BULLET = re.compile(r"^[\u2022\u2023\u25aa\u25e6\u00b7\u2219*\-\u2013\u2014]\s+")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(\.\d{1,2})*\.?|[IVX]{1,5}\.)\s+[A-Z]")


@dataclass
class QuoteMatch:
    verified: bool
    span: tuple[int, int] | None = None
    quote: str | None = None
    method: str = "none"  # exact | fuzzy | none
    score: float = 0.0


def _normalise_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase and collapse whitespace, keeping a map from each output char to its source index."""
    chars: list[str] = []
    index_map: list[int] = []
    previous_space = True
    for index, char in enumerate(text):
        if char.isspace():
            if not previous_space:
                chars.append(" ")
                index_map.append(index)
            previous_space = True
        else:
            lowered = char.lower()
            # Some characters lowercase to several; keep positions aligned with the source.
            for piece in lowered:
                chars.append(piece)
                index_map.append(index)
            previous_space = False
    return "".join(chars), index_map


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END.finditer(text):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def looks_like_prose(quote: str) -> bool:
    """False for table rows such as 'w/o FPH 0.745 0.697 0.638'. They exist in the PDF but state no finding."""
    tokens = quote.split()
    words = [token for token in tokens if re.fullmatch(r"[A-Za-z][A-Za-z'-]+[.,;:)]?", token)]
    return len(words) >= 5 and len(words) / len(tokens) >= 0.5


_PROCEDURE = re.compile(
    r"^\s*(in this (letter|paper|article|work|study),?\s+)?"
    r"(we|our [\w -]{1,30}? (is|are|was|were)|the [\w -]{1,30}? (is|are|was|were))\s+"
    r"(also |then |further |first |here )?"
    r"(evaluat|compar|test|train|use|used|using|conduct|perform|adopt|implement|appl|run|ran|set|select|measur|provid|"
    r"report|describ|present|propos|introduc|stud|investigat|summari|discuss|examin|explor|consider)"
    r"|^\s*(this|these|the) [\w -]{1,30}? (paper |study |work )?(summari[sz]e|describe|present|outline|stud|propose|introduce|investigate)",
    re.I,
)
_OUTCOME = re.compile(
    r"outperform|improv|\bhigher\b|\blower\b|\bbetter\b|\bworse\b|increas|decreas|reduc(e|es|ed|ing|tion)\b|achiev|\breach|"
    r"\bgain(s|ed)?\b|\bdrop(s|ped)?\b|\bfell\b|\brose\b|exceed|superior|significant|\d\s*%|\d\s*percent|"
    r"\bshow(s|ed|n)?\b|\bfound\b|\bfind\b|demonstrat|\brobust\b",
    re.I,
)


_CAPTION = re.compile(r"^\s*(table|fig(ure)?\.?|algorithm)\s*\d+\s*[.:]", re.I)


def is_procedure(quote: str) -> bool:
    """True for captions, and for sentences that describe the study, such as 'We evaluate on X', with no outcome."""
    return bool(_CAPTION.match(quote)) or (bool(_PROCEDURE.match(quote)) and not _OUTCOME.search(quote))


def is_fragment(quote: str) -> bool:
    """True when the start of the sentence is missing, for example because a table interrupts it."""
    return quote.strip()[:1].islower()


def verify_quote(quote: str, source: str, threshold: float = 95, min_chars: int = 20) -> QuoteMatch:
    """Find quote in source. Short quotes are rejected: they would match almost anything."""
    if not quote or not source:
        return QuoteMatch(False)
    normal_quote, _ = _normalise_with_map(quote.strip())
    if len(normal_quote) < min_chars or not looks_like_prose(quote):
        return QuoteMatch(False)

    normal_source, index_map = _normalise_with_map(source)
    position = normal_source.find(normal_quote)
    if position >= 0:
        start, end = whole_sentence(source, index_map[position], index_map[position + len(normal_quote) - 1] + 1)
        return QuoteMatch(True, (start, end), source[start:end], "exact", 100.0)

    best_score = 0.0
    best_span: tuple[int, int] | None = None
    sentences = _sentence_spans(source)
    length = len(normal_quote)
    for first in range(len(sentences)):
        for last in range(first, min(first + 3, len(sentences))):
            window_start, window_end = sentences[first][0], sentences[last][1]
            window = source[window_start:window_end]
            if len(window) < length * 0.8:
                continue
            alignment = fuzz.partial_ratio_alignment(normal_quote, _normalise_with_map(window)[0])
            if alignment is not None and alignment.score > best_score:
                window_norm, window_map = _normalise_with_map(window)
                best_score = alignment.score
                begin = min(alignment.dest_start, len(window_map) - 1)
                finish = min(max(alignment.dest_end - 1, begin), len(window_map) - 1)
                best_span = (window_start + window_map[begin], window_start + window_map[finish] + 1)
            if len(window) > length * 1.5:
                break

    if best_span is not None and best_score >= threshold:
        start, end = best_span
        # Two sentences can be 98% alike and still disagree on the one thing that matters.
        if _numbers(quote) != _numbers(source[start:end]):
            return QuoteMatch(False, score=best_score)
        start, end = whole_sentence(source, start, end)
        return QuoteMatch(True, (start, end), source[start:end], "fuzzy", best_score)
    return QuoteMatch(False, score=best_score)


def _is_heading_line(line: str) -> bool:
    """True for '3 Results' or 'Experimental Results'. False for a sentence cut by a block break."""
    line = line.strip()
    if not line or len(line.split()) > 12 or line[-1] in ".!?,;:":
        return False
    if _NUMBERED_HEADING.match(line):
        return True
    return all(word[0].isupper() for word in line.split() if len(word) > 3) and line[0].isupper()


def _skip_headings(source: str, start: int, end: int) -> int:
    """Move the start of a span past any heading lines the model copied into its quote."""
    while True:
        line_end = source.find("\n", start, end)
        if line_end < 0 or not _is_heading_line(source[start:line_end]):
            return start
        following = line_end
        while following < end and source[following].isspace():
            following += 1
        if following >= end:
            return start
        start = following


def _without_bullet(source: str, start: int, end: int) -> tuple[int, int]:
    found = _BULLET.match(source[start:end])
    return (start + found.end(), end) if found else (start, end)


def whole_sentence(source: str, start: int, end: int, limit: int = 500) -> tuple[int, int]:
    """Widen a partial quote to its full sentence. A fragment can hide what the sentence really says."""
    start = _skip_headings(source, start, end)
    spans = _sentence_spans(source)
    for index, (first, last) in enumerate(spans):
        if first <= start < last:
            if end <= last and last - first <= limit:
                # A block or page break can cut a sentence. Rejoin when the earlier part is unfinished prose.
                if source[first:last].lstrip()[:1].islower() and index > 0:
                    previous = source[spans[index - 1][0] : spans[index - 1][1]].rstrip()
                    if previous[-1:] not in (".", "!", "?", ":") and looks_like_prose(previous) and last - spans[index - 1][0] <= limit:
                        first = spans[index - 1][0]
                # Ignore whitespace at the edges of the sentence span.
                text = source[first:last]
                return _without_bullet(source, first + len(text) - len(text.lstrip()), first + len(text.rstrip()))
            break
    return _without_bullet(source, start, end)


def _numbers(text: str) -> list[str]:
    """The numbers in a text, in order. PDF text can read '9. 2%' where the sentence said '9.2%'."""
    joined = re.sub(r"(?<=\d)[.,]\s+(?=\d)", ".", text)
    return re.findall(r"\d+(?:[.,]\d+)*", joined)


def page_for_offset(offset: int, page_offsets: list[int] | None) -> int | None:
    """Page number (1-based) containing a character offset in the paper's full text."""
    if not page_offsets:
        return None
    return bisect_right(page_offsets, offset)
