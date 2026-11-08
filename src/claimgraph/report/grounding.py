"""Hold the report to the stored relations.

A citation check only proves that a claim exists. The real model wrote "the papers conflict" about
pairs stored as different_setting, each with valid citations. So a sentence stays in a section only if
a stored pair with that relation backs it, and the sections are assembled here rather than by the model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .citation_check import _CITATION, _sentences, extract_ids

HEADINGS = {
    "supports": "Where the papers agree",
    "contradicts": "Where the papers conflict",
    "different_setting": "Where the settings differ",
    "unclear": "Still unclear",
}
_RELATION_OF = {heading.lower(): relation for relation, heading in HEADINGS.items()}
MAX_UNCLEAR_LINES = 8
_FIELD_NAMES = {"dataset": "dataset", "model": "model or population", "metric": "metric", "setting": "setting"}


@dataclass
class Grounded:
    sentences: dict[str, list[str]] = field(default_factory=dict)
    covered: set[frozenset] = field(default_factory=set)
    removed: list[dict] = field(default_factory=list)


def _pair_keys(relations: list[dict]) -> dict[str, set[frozenset]]:
    keys: dict[str, set[frozenset]] = {}
    for pair in relations:
        keys.setdefault(pair["relation"], set()).add(frozenset((pair["claim_a"], pair["claim_b"])))
    return keys


def _backing_pairs(ids: list[str], stored: set[frozenset]) -> list[frozenset]:
    unique = sorted(set(ids))
    return [frozenset((a, b)) for index, a in enumerate(unique) for b in unique[index + 1 :] if frozenset((a, b)) in stored]


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def ground_sentences(text: str, claims: list[dict], relations: list[dict]) -> Grounded:
    """Sort the model's sentences into sections, keeping only those the stored data backs."""
    stored = _pair_keys(relations)
    by_id = {claim["claim_id"]: claim for claim in claims}
    result = Grounded()
    seen: set[str] = set()
    current: str | None = None

    def drop(sentence: str, reason: str) -> None:
        result.removed.append({"sentence": sentence, "reason": reason})

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            current = _RELATION_OF.get(stripped.lstrip("# ").rstrip(":").lower())
            continue
        body = re.sub(r"^([-*]\s+|\d+\.\s+)", "", stripped)
        for sentence in _sentences(body):
            ids = extract_ids(sentence)
            normal = " ".join(sentence.lower().split())
            if normal in seen:
                drop(sentence, "repeated")
                continue
            if current is None:
                drop(sentence, "not under a known heading")
                continue
            if current == "different_setting":
                drop(sentence, "settings are written from the stored conditions")
                continue
            backing = _backing_pairs(ids, stored.get(current, set()))
            if current in ("supports", "contradicts") and not backing:
                drop(sentence, f"no stored {current} pair between the cited claims")
                continue
            if current == "unclear" and len(set(ids)) > 1 and not backing:
                drop(sentence, "no stored unclear pair between the cited claims")
                continue
            cited = " ".join(str(by_id[item].get("text", "")) for item in ids if item in by_id)
            invented = _numbers(_CITATION.sub(" ", sentence)) - _numbers(cited)
            if invented:
                drop(sentence, f"number not in the cited claims: {', '.join(sorted(invented))}")
                continue
            seen.add(normal)
            result.sentences.setdefault(current, []).append(sentence)
            result.covered.update(backing)
    return result


def _short(claim: dict) -> str:
    return claim["text"].rstrip(".")


def _citations(*claims: dict) -> str:
    return " ".join(f"[{claim['claim_id']}]" for claim in claims)


def _stored_sentence(pair: dict, a: dict, b: dict) -> str:
    relation = pair["relation"]
    if relation == "different_setting":
        parts = []
        for name in pair.get("differing_fields") or []:
            first = (a.get("conditions") or {}).get(name) or "not stated"
            second = (b.get("conditions") or {}).get(name) or "not stated"
            parts.append(f"{_FIELD_NAMES.get(name, name)}: {first} in the first, {second} in the second")
        detail = "; ".join(parts) or "the stated conditions differ"
        return f'"{_short(a)}" and "{_short(b)}" were measured differently ({detail}). {_citations(a, b)}'
    joiner = {"supports": "agrees with", "contradicts": "conflicts with"}.get(relation, "cannot yet be compared with")
    reason = pair.get("reason")
    extra = f" ({reason.rstrip('.')})" if reason and relation != "unclear" else ""
    return f'"{_short(a)}" {joiner} "{_short(b)}"{extra}. {_citations(a, b)}'


def compose_report(claims: list[dict], relations: list[dict], grounded: Grounded | None = None) -> str:
    """Assemble the sections. A stored relation the model skipped is still reported, in plain stored terms."""
    grounded = grounded or Grounded()
    by_id = {claim["claim_id"]: claim for claim in claims}
    lines: list[str] = []
    for relation, heading in HEADINGS.items():
        sentences = list(grounded.sentences.get(relation, []))
        for pair in relations:
            key = frozenset((pair["claim_a"], pair["claim_b"]))
            if pair["relation"] != relation or key in grounded.covered:
                continue
            if pair["claim_a"] not in by_id or pair["claim_b"] not in by_id:
                continue
            if relation == "unclear" and len(sentences) >= MAX_UNCLEAR_LINES:
                break
            sentences.append(_stored_sentence(pair, by_id[pair["claim_a"]], by_id[pair["claim_b"]]))
        if sentences:
            lines += [f"## {heading}", *sentences, ""]
    if not lines:
        lines.append(f"## {HEADINGS['unclear']}")
        lines.extend(f"{_short(claim)}. [{claim['claim_id']}]" for claim in claims)
    return "\n".join(lines).strip()
