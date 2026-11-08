"""Claim extraction: section chunks -> LCEL chain -> verified, stored claims."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from ..config import get_settings
from ..llm import check_llm, get_chat_model, structured
from ..models_db import ClaimRow, Paper
from .schema import Claim, Conditions, ExtractedClaim, ExtractedClaimList
from .verify_quote import is_fragment, is_procedure, page_for_offset, verify_quote, whole_sentence

log = logging.getLogger(__name__)
PROMPT_VERSION = "extract_claims.v1"


class LLMUnavailable(RuntimeError):
    """The language model cannot run. Stop the job and keep progress, rather than skip chunks."""


def load_prompt(name: str) -> ChatPromptTemplate:
    raw = (get_settings().prompts_dir / f"{name}.txt").read_text(encoding="utf-8")
    system, human = raw.split("---HUMAN---", 1)
    return ChatPromptTemplate.from_messages([("system", system.strip()), ("human", human.strip())])


def build_extract_chain(llm=None):
    settings = get_settings()
    model = llm if llm is not None else get_chat_model("main")
    attempts = int(settings.models["llm"].get("max_retries", 3))
    chain = load_prompt(PROMPT_VERSION) | structured(model, ExtractedClaimList)
    return chain.with_retry(stop_after_attempt=attempts).with_config(
        metadata={"chain_name": "extract_claims", "prompt_version": PROMPT_VERSION}
    )


@dataclass
class Chunk:
    section: str
    text: str
    char_start: int  # offset in the paper's full text


def merge_sections(sections: list) -> list[tuple[str, str, int]]:
    """Join neighbouring rows of one kind, so that small subsections share a chunk."""
    merged: list[list] = []
    for section in sorted(sections, key=lambda item: item.char_start):
        if merged and merged[-1][0] == section.name and merged[-1][2] + len(merged[-1][1]) == section.char_start:
            merged[-1][1] += section.text
        else:
            merged.append([section.name, section.text, section.char_start])
    return [(name, text, start) for name, text, start in merged]


def make_chunks(sections: list) -> list[Chunk]:
    """Chunks for extraction, most informative sections first, bounded per paper. Order is stable."""
    config = get_settings().pipeline["extraction"]
    wanted = list(config["sections"])
    merged = merge_sections(sections)
    chosen = [item for item in merged if item[0] in wanted]
    if sum(len(item[1]) for item in chosen) < 500:
        # Heading detection found little: fall back to everything except the references.
        chosen = [item for item in merged if item[0] not in ("references", "acknowledgements", "keywords")]

    priority = {name: index for index, name in enumerate(["abstract", "conclusion", "discussion", "results"])}
    chosen.sort(key=lambda item: (priority.get(item[0], 9), item[2]))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(config["chunk_chars"]), chunk_overlap=int(config["chunk_overlap"]), add_start_index=True
    )
    chunks: list[Chunk] = []
    for name, text, start in chosen:
        for piece in splitter.create_documents([text]):
            if len(piece.page_content.strip()) >= 80:
                chunks.append(Chunk(name, piece.page_content, start + int(piece.metadata["start_index"])))
    return chunks[: int(config["max_chunks_per_paper"])]


def to_claim(
    extracted: ExtractedClaim,
    paper: Paper,
    chunk: Chunk,
    number: int,
    threshold: float,
    min_chars: int,
    full_text: str | None = None,
) -> Claim | None:
    """Verify the quote against the chunk the model saw. Returns None if it is not really there."""
    match = verify_quote(extracted.quote, chunk.text, threshold=threshold, min_chars=min_chars)
    if not match.verified or match.span is None:
        return None
    start = chunk.char_start + match.span[0]
    end = chunk.char_start + match.span[1]
    quote = match.quote or ""
    if full_text:
        # A chunk can begin mid-sentence, so the sentence is completed from the whole paper.
        start, end = whole_sentence(full_text, start, end)
        quote = full_text[start:end]
    if is_procedure(quote) or is_fragment(quote):
        return None
    return Claim(
        claim_id=f"{paper.id}_c{number}",
        paper_id=paper.id,
        text=extracted.text.strip(),
        intervention=extracted.intervention.strip(),
        outcome=extracted.outcome.strip(),
        direction=extracted.direction,
        conditions=extracted.conditions or Conditions(),
        hedged=extracted.hedged,
        section=chunk.section,
        page=page_for_offset(start, paper.page_offsets),
        quote=" ".join(quote.split()),
        quote_span=(start, end),
        quote_verified=True,
    )


def claim_to_row(claim: Claim) -> ClaimRow:
    return ClaimRow(
        claim_id=claim.claim_id,
        paper_id=claim.paper_id,
        text=claim.text,
        intervention=claim.intervention,
        outcome=claim.outcome,
        direction=claim.direction,
        conditions=claim.conditions.model_dump(),
        hedged=claim.hedged,
        section=claim.section,
        page=claim.page,
        quote=claim.quote,
        quote_span_start=claim.quote_span[0] if claim.quote_span else None,
        quote_span_end=claim.quote_span[1] if claim.quote_span else None,
        quote_verified=claim.quote_verified,
    )


def extract_paper(
    session: Session,
    paper: Paper,
    chain=None,
    on_chunk: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """Extract one paper, committing after every chunk so that an interrupted run can resume."""
    settings = get_settings()
    chain = chain if chain is not None else build_extract_chain()
    verify = settings.pipeline["quote_verify"]
    threshold, min_chars = float(verify["fuzzy_threshold"]), int(verify.get("min_quote_chars", 20))

    chunks = make_chunks(paper.sections)
    full_text = "".join(section.text for section in sorted(paper.sections, key=lambda item: item.char_start))
    paper.chunks_total = len(chunks)
    session.commit()

    stats = {"proposed": 0, "kept": 0, "dropped": 0, "chunks_failed": 0}
    numbers = [int(row.claim_id.rsplit("_c", 1)[1]) for row in paper.claims]
    next_number = max(numbers, default=0) + 1
    seen = {(row.quote_span_start, row.quote_span_end) for row in paper.claims}
    seen_quotes = {" ".join(row.quote.lower().split()) for row in paper.claims}

    for index, chunk in enumerate(chunks):
        if index < paper.chunks_done:
            continue
        if should_cancel and should_cancel():
            return {**stats, "cancelled": True}
        try:
            output = chain.invoke({"title": paper.title or "", "section": chunk.section, "text": chunk.text})
        except Exception as exc:  # noqa: BLE001
            status = check_llm()
            if status.get("fatal"):
                raise LLMUnavailable(status.get("detail", "The language model cannot run.")) from exc
            log.warning("Chunk %d of %s failed after retries: %s", index, paper.id, exc)
            stats["chunks_failed"] += 1
            output = ExtractedClaimList()

        for extracted in output.claims if output else []:
            stats["proposed"] += 1
            claim = to_claim(extracted, paper, chunk, next_number, threshold, min_chars, full_text)
            if claim is None:
                stats["dropped"] += 1
                paper.claims_dropped = (paper.claims_dropped or 0) + 1
                continue
            normal_quote = " ".join(claim.quote.lower().split())
            if claim.quote_span in seen or normal_quote in seen_quotes:
                continue  # the same sentence again, from the chunk overlap
            seen.add(claim.quote_span)
            seen_quotes.add(normal_quote)
            session.add(claim_to_row(claim))
            next_number += 1
            stats["kept"] += 1

        paper.chunks_done = index + 1
        session.commit()
        if on_chunk:
            on_chunk(index + 1, len(chunks))

    paper.extract_status = "done"
    session.commit()
    _log_extraction(paper.id, stats)
    return stats


def _log_extraction(paper_id: str, stats: dict) -> None:
    try:
        settings = get_settings()
        settings.logs_dir.mkdir(parents=True, exist_ok=True)
        with (settings.logs_dir / "extraction.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"paper_id": paper_id, **stats}) + "\n")
    except OSError:
        log.debug("Could not write the extraction log", exc_info=True)
