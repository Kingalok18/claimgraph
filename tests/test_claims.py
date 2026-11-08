import json

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from fakes import FakeChatStructured, fake_extract_chain
from claimgraph.claims.extract_chain import build_extract_chain, extract_paper, load_prompt, make_chunks, merge_sections
from claimgraph.claims.schema import Claim, ClaimList, Conditions, ExtractedClaim, ExtractedClaimList
from claimgraph.claims.verify_quote import is_fragment, is_procedure, page_for_offset, verify_quote, whole_sentence
from claimgraph.ingest.library import scan_library
from claimgraph.models_db import ClaimRow, Paper


SOURCE = (
    "We trained the network for ten epochs. Adding dropout increased the test accuracy from 92.1 percent to 93.4 percent. "
    "The difference was consistent across seeds."
)


def test_schema_requires_known_direction():
    with pytest.raises(ValidationError):
        ExtractedClaim(text="t", intervention="i", outcome="o", direction="up", quote="q")


def test_conditions_blank_values_become_none():
    conditions = Conditions(dataset="", model="N/A", metric="null", setting="Unknown")
    assert conditions.model_dump() == {"dataset": None, "model": None, "metric": None, "setting": None}
    assert Conditions(dataset="CIFAR-10").dataset == "CIFAR-10"


def test_claimlist_round_trip():
    claim = Claim(
        claim_id="p1_c1", paper_id="p1", text="t", intervention="i", outcome="o", direction="decrease",
        conditions=Conditions(), hedged=True, section="results", quote="q",
    )
    data = json.loads(ClaimList(claims=[claim]).model_dump_json())
    assert data["claims"][0]["quote_verified"] is False


def test_llm_cannot_set_code_owned_fields():
    fields = set(ExtractedClaim.model_fields)
    assert not fields & {"claim_id", "paper_id", "page", "quote_span", "quote_verified", "section"}


def test_exact_quote_is_verified_with_span():
    quote = "Adding dropout increased the test accuracy from 92.1 percent to 93.4 percent."
    match = verify_quote(quote, SOURCE)
    assert match.verified and match.method == "exact"
    assert SOURCE[match.span[0] : match.span[1]] == quote


def test_whitespace_and_case_differences_are_exact():
    match = verify_quote("adding  dropout\nincreased the TEST accuracy", SOURCE)
    assert match.verified and match.method == "exact"
    assert match.quote.startswith("Adding dropout")


def test_small_typo_passes_fuzzy_and_stores_the_papers_own_words():
    quote = "Adding dropout increased the test acuracy from 92.1 percent to 93.4 percent."
    match = verify_quote(quote, SOURCE)
    assert match.verified and match.method == "fuzzy"
    assert "accuracy" in match.quote and "acuracy" not in match.quote


@pytest.mark.parametrize(
    "quote",
    [
        "Adding dropout decreased the validation loss of the model by a large margin overall.",
        "This sentence is simply not in the paper at all, in any form.",
    ],
)
def test_invented_quote_is_rejected(quote):
    assert verify_quote(quote, SOURCE).verified is False


def test_changed_number_is_rejected():
    """A wrong figure is the most damaging kind of misquote."""
    long_source = "Accuracy was measured on the held-out test set of the benchmark and reached 12 percent overall in this study."
    quote = "Accuracy was measured on the held-out test set of the benchmark and reached 97 percent overall in this study."
    assert verify_quote(quote, long_source).verified is False


def test_pdf_spacing_in_a_number_is_not_a_changed_number():
    source = "Our method outperforms the previous best by 9. 2% in terms of F-measure on the test set of the benchmark."
    quote = "Our method outperforms the previous best by 9.2% in terms of F-measure on the test set of the benchmark."
    assert verify_quote(quote, source).verified is True


@pytest.mark.parametrize(
    "row",
    [
        "w/o FPH 0.745 0.697 0.638 0.666 0.528 0.649 0.588 0.617 0.576 0.626 0.653 0.639",
        "DTD (Ours) 0.820 0.814 0.843 0.828",
        "Method Dataset IoU P R F",
    ],
)
def test_table_row_is_rejected_even_though_it_is_in_the_pdf(row):
    assert verify_quote(row, f"Some text before. {row} Some text after.").verified is False


def test_prose_with_many_numbers_is_kept():
    quote = "Our proposed DTD outperforms previous methods by 9.2%, 26.3% and 12.3% in terms of F-measure on the testing sets."
    assert verify_quote(quote, f"Intro. {quote} More.").verified is True


@pytest.mark.parametrize(
    "quote",
    [
        "Our models are evaluated on the testing set of the DocTamper dataset and the public T-SROIE dataset.",
        "We evaluate the tampered text detection performance on the DocTamper dataset.",
        "We compare the tampered text detection performance of our models on the DocTamper dataset.",
        "We also trained the network for ten epochs with a batch size of 128.",
    ],
)
def test_procedure_sentences_are_not_findings(quote):
    assert is_procedure(quote) is True


@pytest.mark.parametrize(
    "quote",
    [
        "We describe the training procedure and report the measured test accuracy.",
        "The measurements above summarise our study of dropout in residual networks.",
        "This paper studies the effect of dropout regularisation on ResNet-50.",
        "We propose a new network for document forgery localisation.",
    ],
)
def test_statements_about_the_paper_are_not_findings(quote):
    assert is_procedure(quote) is True


def test_partial_quote_widens_to_its_sentence():
    source = "Intro sentence here. We describe the training procedure and report the measured test accuracy. Next one."
    match = verify_quote("report the measured test accuracy.", source)
    assert match.verified and match.quote == "We describe the training procedure and report the measured test accuracy."
    assert source[match.span[0] : match.span[1]] == match.quote


HEADED_SOURCE = (
    "2 Method\nWe trained each network for 200 epochs.\n\n"
    "3 Results\nAdding dropout increased the test accuracy of ResNet-50 on CIFAR-10.\nThe difference was consistent.\n\n"
    "4 Conclusion\nThe measurements above summarise our study of dropout in residual networks.\n"
)


@pytest.mark.parametrize(
    "quote",
    [
        "Adding dropout increased the test accuracy of ResNet-50 on CIFAR-10.",
        "increased the test accuracy of ResNet-50 on CIFAR-10",
        "3 Results Adding dropout increased the test accuracy of ResNet-50 on CIFAR-10.",  # the model copied the heading
    ],
)
def test_section_heading_never_joins_the_quote(quote):
    """Seen with the real model: stored quotes began '3 Results Adding dropout ...'."""
    match = verify_quote(quote, HEADED_SOURCE)
    assert match.verified and match.quote == "Adding dropout increased the test accuracy of ResNet-50 on CIFAR-10."
    assert HEADED_SOURCE[match.span[0] : match.span[1]] == match.quote


def test_heading_cannot_hide_a_procedure_sentence():
    """A heading in front of the sentence used to carry it past the procedure filter."""
    match = verify_quote("The measurements above summarise our study of dropout in residual networks.", HEADED_SOURCE)
    assert match.verified and is_procedure(match.quote) is True


@pytest.mark.parametrize(
    "quote",
    [
        "Table 3. Ablation study on DocTamper dataset. All images are compressed randomly one to three times.",
        "Fig. 2: Results improve with more training data on the benchmark.",
        "In this letter, we propose a novel document image forgery localization network named TSFE-Net.",
        "The dataset used in this article is provided by the fifth forgery identification competition.",
    ],
)
def test_captions_and_paper_descriptions_seen_in_real_papers_are_not_findings(quote):
    assert is_procedure(quote) is True


def test_bullet_is_not_part_of_the_quote():
    source = "Summary of results.\n\u2022 The multimodal system yields the lowest FPR, demonstrating strong robustness.\n"
    match = verify_quote("The multimodal system yields the lowest FPR, demonstrating strong robustness.", source)
    assert match.quote == "The multimodal system yields the lowest FPR, demonstrating strong robustness."


def test_sentence_broken_by_a_block_is_rejoined():
    source = "Unlike earlier work on forged document images,\n\nthis paper locates the forged text area in each image.\n"
    start = source.index("this paper")
    first, last = whole_sentence(source, start, start + 30)
    assert source[first:last].startswith("Unlike earlier") and source[first:last].endswith("image.")


def test_sentence_interrupted_by_a_table_stays_a_fragment():
    """Seen in a real paper: the sentence's start is lost, so the claim is dropped rather than stored half-read."""
    source = "EAST [52] 0.9191 0.8960 0.9075 ATRR [45] 0.9471 0.9249\n\nwithout any of the proposed parts serves as the baseline model here.\n"
    match = verify_quote("without any of the proposed parts serves as the baseline model here.", source)
    assert match.verified and is_fragment(match.quote)


def test_chunk_that_starts_mid_sentence_is_completed_from_the_whole_paper(fixture_papers, session):
    from claimgraph.claims.extract_chain import Chunk, to_claim
    from claimgraph.models_db import Paper

    full = "Intro here. Compared with earlier methods, the algorithm in this paper locates the forged text area. End."
    position = full.index("the algorithm")
    paper = Paper(id="p1", file_path="x.pdf", file_hash="h", source="user", title="t")
    extracted = ExtractedClaim(
        text="t", intervention="i", outcome="o", direction="increase",
        quote="the algorithm in this paper locates the forged text area.",
    )
    chunk = Chunk("results", full[position:], position)
    claim = to_claim(extracted, paper, chunk, 1, 95, 20, full)
    assert claim.quote == "Compared with earlier methods, the algorithm in this paper locates the forged text area."
    assert to_claim(extracted, paper, chunk, 1, 95, 20) is None  # without the paper, it stays a fragment


def test_sentence_cut_by_a_block_break_keeps_both_parts():
    source = "Intro sentence goes here.\n\nWe trained the model on\n\nImageNet and reached 70 percent accuracy overall.\n"
    match = verify_quote("We trained the model on ImageNet and reached 70 percent accuracy overall.", source)
    assert match.verified
    assert " ".join(match.quote.split()) == "We trained the model on ImageNet and reached 70 percent accuracy overall."


@pytest.mark.parametrize(
    "quote",
    [
        "We compare our model with ten others and it outperforms all of them on F-measure.",
        "Our model is evaluated on DocTamper, where it reaches 82 percent.",
        "Adding dropout increased the test accuracy of ResNet-50 on CIFAR-10.",
        "Experiments demonstrate that our proposed DTD outperforms previous methods by 9.2%.",
    ],
)
def test_results_are_not_mistaken_for_procedure(quote):
    assert is_procedure(quote) is False


def test_short_quote_is_rejected():
    assert verify_quote("Adding dropout", SOURCE).verified is False


@pytest.mark.parametrize("quote,source", [("", SOURCE), (SOURCE, "")])
def test_empty_inputs(quote, source):
    assert verify_quote(quote, source).verified is False


def test_offsets_map_to_pages():
    offsets = [0, 100, 250]
    assert page_for_offset(0, offsets) == 1
    assert page_for_offset(99, offsets) == 1
    assert page_for_offset(100, offsets) == 2
    assert page_for_offset(9999, offsets) == 3
    assert page_for_offset(5, None) is None


class Row:
    def __init__(self, name, text, char_start):
        self.name, self.text, self.char_start = name, text, char_start


def test_neighbouring_sections_merge():
    rows = [Row("results", "aaa", 0), Row("results", "bb", 3), Row("method", "c", 5), Row("results", "d", 6)]
    assert merge_sections(rows) == [("results", "aaabb", 0), ("method", "c", 5), ("results", "d", 6)]


def test_chunks_use_configured_sections_with_valid_offsets():
    text_a = "Abstract text about a finding. " * 10
    text_b = "Method detail that is long. " * 40
    text_c = "Result sentence with numbers. " * 30
    rows = [Row("abstract", text_a, 0), Row("method", text_b, len(text_a)), Row("results", text_c, len(text_a) + len(text_b))]
    full = text_a + text_b + text_c
    chunks = make_chunks(rows)
    assert {chunk.section for chunk in chunks} == {"abstract", "results"}
    for chunk in chunks:
        assert full[chunk.char_start : chunk.char_start + len(chunk.text)] == chunk.text


def test_chunks_fall_back_when_headings_were_not_found():
    rows = [Row("other", "Body text of a paper with odd headings. " * 30, 0), Row("references", "[1] Someone. " * 50, 1200)]
    chunks = make_chunks(rows)
    assert chunks and {chunk.section for chunk in chunks} == {"other"}


def test_chunks_are_bounded_per_paper():
    rows = [Row("results", "Long results paragraph here. " * 2000, 0)]
    assert len(make_chunks(rows)) <= 12


def test_prompt_template_needs_only_declared_inputs():
    assert set(load_prompt("extract_claims.v1").input_variables) == {"title", "section", "text"}


def test_chain_returns_structured_list():
    value = ExtractedClaimList(claims=[])
    chain = build_extract_chain(llm=FakeChatStructured(value))
    assert chain.invoke({"title": "t", "section": "results", "text": "x"}) == value


def _paper(session, papers_dir, name_part: str):
    scan_library(session, papers_dir)
    papers = session.scalars(select(Paper).options(selectinload(Paper.sections), selectinload(Paper.claims))).all()
    return next(paper for paper in papers if name_part in paper.file_path)


def test_extraction_stores_verified_claims_with_page(fixture_papers, session):
    paper = _paper(session, fixture_papers, "fixture_a")
    stats = extract_paper(session, paper, chain=fake_extract_chain())
    assert stats["kept"] == 1 and stats["dropped"] == 0
    row = session.scalar(select(ClaimRow))
    assert row.claim_id == f"{paper.id}_c1"
    assert row.quote_verified and row.page == 1 and row.section == "results"
    assert "increased the test accuracy" in row.quote
    assert paper.extract_status == "done"
    full = "".join(section.text for section in sorted(paper.sections, key=lambda s: s.char_start))
    assert " ".join(full[row.quote_span_start : row.quote_span_end].split()) == row.quote


def test_invented_quotes_are_dropped_and_counted(fixture_papers, session):
    paper = _paper(session, fixture_papers, "fixture_a")
    stats = extract_paper(session, paper, chain=fake_extract_chain(extra_bad_quote=True))
    assert stats["kept"] == 1 and stats["dropped"] == 1
    assert paper.claims_dropped == 1
    assert session.scalar(select(func.count()).select_from(ClaimRow).where(ClaimRow.quote_verified.is_(False))) == 0


def test_interrupted_extraction_resumes(fixture_papers, session):
    paper = _paper(session, fixture_papers, "fixture_a")
    calls = {"count": 0}

    def cancel_after_one():
        calls["count"] += 1
        return calls["count"] > 1

    stats = extract_paper(session, paper, chain=fake_extract_chain(), should_cancel=cancel_after_one)
    assert stats.get("cancelled") and paper.extract_status == "pending"
    done = paper.chunks_done
    assert done == 1
    extract_paper(session, paper, chain=fake_extract_chain())
    assert paper.extract_status == "done" and paper.chunks_done == paper.chunks_total
    # The planted sentence is stored once however the run was split.
    assert session.scalar(select(func.count()).select_from(ClaimRow)) == 1


def test_failed_chunk_is_counted_and_the_paper_continues(fixture_papers, session, monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from claimgraph.claims import extract_chain as module

    monkeypatch.setattr(module, "check_llm", lambda: {"fatal": False})
    paper = _paper(session, fixture_papers, "fixture_a")

    def boom(_inputs):
        raise ValueError("bad output")

    stats = extract_paper(session, paper, chain=RunnableLambda(boom))
    assert stats["chunks_failed"] == paper.chunks_total and paper.extract_status == "done"


def test_unavailable_model_stops_without_losing_chunks(fixture_papers, session, monkeypatch):
    from langchain_core.runnables import RunnableLambda

    from claimgraph.claims import extract_chain as module

    monkeypatch.setattr(module, "check_llm", lambda: {"fatal": True, "detail": "gone"})
    paper = _paper(session, fixture_papers, "fixture_a")

    def boom(_inputs):
        raise ConnectionError("down")

    with pytest.raises(module.LLMUnavailable):
        extract_paper(session, paper, chain=RunnableLambda(boom))
    assert paper.chunks_done == 0 and paper.extract_status == "pending"
