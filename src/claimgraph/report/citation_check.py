"""Keep only report sentences whose citations all resolve to existing claims.

A sentence with an unknown id, or a factual sentence with no citation, is removed and logged.
Markdown headings are kept: they state no finding.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[([^\[\]]+)\]")
_SENTENCE = re.compile(r"(?<=[.!?\]])\s+(?=[A-Z\[\"'(])")
_ONLY_CITATIONS = re.compile(r"(?:\[[^\[\]]+\]\s*)+\.?")


@dataclass
class CitationResult:
    text: str
    cited: list[str] = field(default_factory=list)
    removed: list[dict] = field(default_factory=list)


def extract_ids(sentence: str) -> list[str]:
    ids: list[str] = []
    for group in _CITATION.findall(sentence):
        for part in re.split(r"[,;\s]+", group.strip()):
            if part:
                ids.append(part)
    return ids


def _check_sentence(sentence: str, known: set[str]) -> tuple[bool, str, list[str]]:
    ids = extract_ids(sentence)
    if not ids:
        return False, "no citation", []
    unknown = [item for item in ids if item not in known]
    if unknown:
        return False, f"unknown id: {', '.join(unknown)}", ids
    return True, "", ids


def _sentences(body: str) -> list[str]:
    """Split into sentences. Citations split off on their own belong to the sentence before them."""
    merged: list[str] = []
    for piece in _SENTENCE.split(body):
        piece = piece.strip()
        if not piece:
            continue
        if merged and _ONLY_CITATIONS.fullmatch(piece):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def check_citations(report: str, known_ids: set[str] | list[str]) -> CitationResult:
    known = set(known_ids)
    kept_lines: list[str] = []
    cited: list[str] = []
    removed: list[dict] = []

    for line in report.splitlines():
        stripped = line.strip()
        if not stripped:
            kept_lines.append("")
            continue
        if stripped.startswith("#"):
            kept_lines.append(stripped)
            continue
        prefix = ""
        body = stripped
        bullet = re.match(r"^([-*]\s+|\d+\.\s+)", stripped)
        if bullet:
            prefix, body = bullet.group(1), stripped[bullet.end() :]
        kept: list[str] = []
        for sentence in _sentences(body):
            ok, reason, ids = _check_sentence(sentence, known)
            if ok:
                kept.append(sentence)
                cited.extend(ids)
            else:
                removed.append({"sentence": sentence, "reason": reason})
                log.info("Removed report sentence (%s): %s", reason, sentence[:120])
        if kept:
            kept_lines.append(prefix + " ".join(kept))

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines)).strip()
    text = _drop_empty_headings(text)
    return CitationResult(text=text, cited=sorted(set(cited)), removed=removed)


def _drop_empty_headings(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    for index, line in enumerate(lines):
        if line.startswith("#"):
            following = next((item for item in lines[index + 1 :] if item.strip()), "")
            if not following or following.startswith("#"):
                continue
        output.append(line)
    return "\n".join(output).strip()
