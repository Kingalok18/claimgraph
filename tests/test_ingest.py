import io
import shutil
from pathlib import Path

import pytest
from langchain_core.documents import Document
from sqlalchemy import func, select

from conftest import FIXTURES
from claimgraph.ingest.grobid_loader import _first_page, documents_to_parsed
from claimgraph.ingest.library import file_sha256, find_pdfs, save_upload, scan_library
from claimgraph.ingest.pymupdf_loader import _join_lines, parse_pdf
from claimgraph.ingest.sections import canonical_name, classify_heading, parent_aware_name
from claimgraph.models_db import Paper, Section


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Experimental Results", "results"),
        ("Results and Discussion", "results"),
        ("Conclusion and Future Work", "conclusion"),
        ("Related Work", "related_work"),
        ("Proposed Method", "method"),
        ("REFERENCES", "references"),
        ("Acknowledgments", "acknowledgements"),
        ("Abstract", "abstract"),
        ("Purple elephants", None),
    ],
)
def test_canonical_names(title, expected):
    assert canonical_name(title) == expected


def test_heading_numbered_needs_prominence():
    assert classify_heading("4 Results", 12, True, 10).canonical == "results"
    assert classify_heading("4 Results", 10, False, 10) is None  # a table row, not a heading


def test_heading_levels_and_inheritance():
    sub = classify_heading("4.2 Metrics", 11, True, 10)
    assert sub.level == 2
    assert parent_aware_name(sub, "results") == "results"
    roman = classify_heading("IV. EXPERIMENTS", 10, True, 10)
    assert roman.level == 1 and roman.canonical == "results"


def test_inline_abstract():
    heading = classify_heading("Abstract—We study forgery detection in documents.", 10, False, 10)
    assert heading.canonical == "abstract"
    assert heading.inline_rest.startswith("We study")


@pytest.mark.parametrize(
    "line",
    ["Figure 3 shows the results of the method", "Table 2: Results", "The results of this experiment show gains on the test set overall."],
)
def test_body_text_is_not_a_heading(line):
    assert classify_heading(line, 10, False, 10) is None


def test_hyphenation_only_joins_real_word_breaks():
    assert _join_lines(["the detec-", "tion rate"]) == "the detection rate"
    assert _join_lines(["the ResNet-", "50 model"]) == "the ResNet- 50 model"


def test_pymupdf_parses_fixture(expected):
    paper = expected["papers"][0]
    parsed = parse_pdf(FIXTURES / paper["file"])
    assert parsed.title == paper["title"]
    assert parsed.n_pages == 1
    names = [section.name for section in parsed.sections]
    for name in ("abstract", "related_work", "method", "results", "conclusion"):
        assert name in names
    results = next(section for section in parsed.sections if section.name == "results")
    assert paper["claims"][0]["quote"] in " ".join(results.text.split())
    assert parsed.abstract.startswith("This paper studies")


def test_sections_partition_the_full_text(expected):
    """Character offsets are only valid if the sections join back into the text exactly."""
    parsed = parse_pdf(FIXTURES / expected["papers"][1]["file"])
    position = 0
    for section in parsed.sections:
        assert section.char_start == position
        position += len(section.text)
    assert position == len(parsed.full_text)


def test_scan_rejects_nothing_silently_for_a_textless_pdf(papers_dir, session):
    from reportlab.pdfgen import canvas

    blank = papers_dir / "scan.pdf"
    pdf = canvas.Canvas(str(blank))
    pdf.showPage()
    pdf.save()
    result = scan_library(session, papers_dir)
    assert result.failed == 1
    row = session.scalar(select(Paper))
    assert row.parse_status == "failed" and "OCR" in row.parse_error


def test_corrupt_pdf_does_not_stop_the_scan(fixture_papers, session):
    (fixture_papers / "broken.pdf").write_bytes(b"not a pdf at all")
    result = scan_library(session, fixture_papers)
    assert result.found == 5 and result.new == 4 and result.failed == 1


def test_scan_imports_and_is_idempotent(fixture_papers, session):
    first = scan_library(session, fixture_papers)
    assert first.new == 4
    sections = session.scalar(select(func.count()).select_from(Section))
    second = scan_library(session, fixture_papers)
    assert second.new == 0
    assert session.scalar(select(func.count()).select_from(Paper)) == 4
    assert session.scalar(select(func.count()).select_from(Section)) == sections


def test_same_file_under_another_name_is_not_duplicated(fixture_papers, session, expected):
    shutil.copy(fixture_papers / expected["papers"][0]["file"], fixture_papers / "copy.pdf")
    assert scan_library(session, fixture_papers).new == 4
    assert session.scalar(select(func.count()).select_from(Paper)) == 4


def test_scan_never_changes_user_files(fixture_papers, session):
    before = {path.name: file_sha256(path) for path in find_pdfs(fixture_papers)}
    scan_library(session, fixture_papers)
    assert {path.name: file_sha256(path) for path in find_pdfs(fixture_papers)} == before


def test_removed_file_leaves_the_library_and_returns(fixture_papers, session, expected):
    scan_library(session, fixture_papers)
    name = expected["papers"][0]["file"]
    gone_id = session.scalar(select(Paper.id).where(Paper.file_path == str(fixture_papers / name)))
    holding = fixture_papers.parent / "held.pdf"
    shutil.move(fixture_papers / name, holding)
    assert scan_library(session, fixture_papers).removed == 1
    assert session.get(Paper, gone_id) is None
    assert session.scalar(select(func.count()).select_from(Paper)) == len(expected["papers"]) - 1
    assert session.scalar(select(func.count()).select_from(Section).where(Section.paper_id == gone_id)) == 0
    assert holding.exists()  # the file itself is never touched
    shutil.move(holding, fixture_papers / name)
    assert scan_library(session, fixture_papers).new == 1
    assert session.get(Paper, gone_id) is not None  # same file, same id


def test_missing_folder_removes_nothing(fixture_papers, session, expected):
    scan_library(session, fixture_papers)
    assert scan_library(session, fixture_papers.parent / "not_there").removed == 0
    assert session.scalar(select(func.count()).select_from(Paper)) == len(expected["papers"])


def test_removing_an_analysed_paper_cleans_up_what_was_built_on_it(papers_dir, session):
    from fakes import FakeEmbeddings

    from claimgraph.embed import embeddings
    from claimgraph.embed.vectorstore import index_claims, indexed_ids
    from claimgraph.models_db import ClaimPair, ClaimRow, ClaimTopic, Label, Report, Topic

    embeddings.set_embeddings(FakeEmbeddings())
    files = {}
    for paper_id in ("pa", "pb", "pc"):
        files[paper_id] = papers_dir / f"{paper_id}.pdf"
        files[paper_id].write_bytes(b"%PDF-" + paper_id.encode())
        session.add(Paper(id=paper_id, file_path=str(files[paper_id]), file_hash=file_sha256(files[paper_id]), parse_status="parsed", extract_status="done"))
    session.flush()

    def claim(claim_id, paper_id):
        return ClaimRow(
            claim_id=claim_id, paper_id=paper_id, text="t", intervention="method " + claim_id, outcome="accuracy",
            direction="increases", conditions={}, section="results", quote="q", quote_verified=True,
        )

    session.add_all([claim("pa_c1", "pa"), claim("pb_c1", "pb"), claim("pc_c1", "pc")])
    shared = Topic(topic_key="shared", label="Shared", source="auto")
    only_a = Topic(topic_key="only_a", label="Only A", source="auto")
    question = Topic(topic_key="question", label="My question", source="question", question_text="Does it work?")
    session.add_all([shared, only_a, question])
    session.flush()
    session.add_all([
        ClaimTopic(claim_id="pa_c1", topic_id=shared.id), ClaimTopic(claim_id="pb_c1", topic_id=shared.id),
        ClaimTopic(claim_id="pc_c1", topic_id=shared.id), ClaimTopic(claim_id="pa_c1", topic_id=only_a.id),
        ClaimTopic(claim_id="pa_c1", topic_id=question.id),
        ClaimPair(pair_key="ab", topic_id=shared.id, claim_a="pa_c1", claim_b="pb_c1", relation="unclear"),
        ClaimPair(pair_key="bc", topic_id=shared.id, claim_a="pb_c1", claim_b="pc_c1", relation="supports"),
        Report(topic_id=shared.id, text="old", cited_claim_ids=["pa_c1"], prompt_version="v1", model="m"),
        Label(kind="pair", target_id="ab", value={"relation": "supports", "predicted": "unclear"}),
    ])
    session.commit()
    index_claims([{"claim_id": c, "paper_id": c[:2], "intervention": "method " + c, "outcome": "accuracy"} for c in ("pa_c1", "pb_c1", "pc_c1")])
    shared_id, only_a_id, question_id = shared.id, only_a.id, question.id

    files["pa"].unlink()
    assert scan_library(session, papers_dir).removed == 1
    session.expire_all()

    assert session.get(Paper, "pa") is None and session.get(ClaimRow, "pa_c1") is None
    assert set(session.scalars(select(ClaimPair.pair_key))) == {"bc"}
    assert set(session.scalars(select(ClaimTopic.claim_id).where(ClaimTopic.topic_id == shared_id))) == {"pb_c1", "pc_c1"}
    assert session.scalar(select(func.count()).select_from(Report)) == 0  # the report cited a removed claim
    assert session.get(Topic, only_a_id) is None  # no claims left
    assert session.get(Topic, question_id) is not None  # the user's own question stays
    assert session.scalar(select(func.count()).select_from(Label)) == 1  # labels are the user's work
    assert indexed_ids(["pa_c1", "pb_c1", "pc_c1"]) == {"pb_c1", "pc_c1"}
    assert files["pb"].exists() and files["pc"].exists()


def test_scan_finds_subfolders_and_uploads(fixture_papers, session, expected):
    sub = fixture_papers / "uploads"
    sub.mkdir()
    shutil.move(fixture_papers / expected["papers"][0]["file"], sub / "moved.pdf")
    assert scan_library(session, fixture_papers).new == 4


def test_upload_name_cannot_escape_the_folder(papers_dir):
    path = save_upload(io.BytesIO(b"%PDF-"), "..\\..\\evil name?.pdf")
    assert path.parent == papers_dir / "uploads"
    second = save_upload(io.BytesIO(b"%PDF-"), "..\\..\\evil name?.pdf")
    assert second != path and second.parent == path.parent


@pytest.mark.parametrize("value,expected_page", [("(3, 4)", 3), ((5, 6), 5), ("7", 7), (None, None)])
def test_grobid_page_parsing(value, expected_page):
    assert _first_page(value) == expected_page


def test_grobid_documents_convert_to_shared_format():
    documents = [
        Document(page_content="We measured it.", metadata={"paper_title": "T", "section_title": "Results", "pages": "(2, 2)"}),
        Document(page_content="Accuracy rose.", metadata={"paper_title": "T", "section_title": "Results", "pages": "(2, 2)"}),
        Document(page_content="It works.", metadata={"paper_title": "T", "section_title": "Conclusion", "pages": "(3, 3)"}),
    ]
    parsed = documents_to_parsed(documents)
    assert parsed.title == "T"
    assert [section.name for section in parsed.sections] == ["results", "conclusion"]
    assert "We measured it. Accuracy rose." in parsed.sections[0].text
    position = 0
    for section in parsed.sections:
        assert section.char_start == position
        position += len(section.text)
    assert parsed.page_offsets[1] == 0 and parsed.page_offsets[2] == parsed.sections[1].char_start


def test_grobid_availability_check_handles_no_server():
    from claimgraph.ingest.grobid_loader import grobid_available

    assert grobid_available("http://127.0.0.1:1", timeout=0.2) is False
