from langchain_core.runnables import RunnableLambda
from sqlalchemy import select

from claimgraph.graph.build import build_graph
from claimgraph.graph.export import PAPER_COLOURS, to_cytoscape
from claimgraph.models_db import ClaimPair, ClaimRow, ClaimTopic, Paper, Topic
from claimgraph.report.citation_check import check_citations, extract_ids
from claimgraph.report.grounding import compose_report, ground_sentences
from claimgraph.report.report_chain import generate_report, latest_report, template_report

KNOWN = {"p1_c1", "p2_c1", "p3_c7"}


def test_extract_ids_handles_forms():
    assert extract_ids("Text [p1_c1] and [p2_c1, p3_c7].") == ["p1_c1", "p2_c1", "p3_c7"]


def test_valid_sentences_are_kept():
    result = check_citations("## Where the papers agree\nDropout helps. [p1_c1] [p2_c1]", KNOWN)
    assert "Dropout helps." in result.text and result.cited == ["p1_c1", "p2_c1"] and not result.removed


def test_unknown_id_removes_only_that_sentence():
    result = check_citations("Real finding [p1_c1]. Invented finding [p9_c9].", KNOWN)
    assert "Real finding" in result.text and "Invented" not in result.text
    assert result.removed[0]["reason"].startswith("unknown id")


def test_uncited_sentence_is_removed():
    result = check_citations("Overall this field is exciting. Real finding [p1_c1].", KNOWN)
    assert "exciting" not in result.text and result.removed[0]["reason"] == "no citation"


def test_one_bad_id_among_good_removes_the_sentence():
    assert "Mixed" not in check_citations("Mixed claim [p1_c1] [zzz_c1].", KNOWN).text


def test_heading_left_empty_is_dropped():
    result = check_citations("## Empty\nNo citation here.\n## Kept\nFinding [p1_c1].", KNOWN)
    assert "## Empty" not in result.text and "## Kept" in result.text


def test_citation_after_full_stop_stays_with_its_sentence():
    result = check_citations("Dropout helps. [p1_c1]\n- Bullet finding [p2_c1].", KNOWN)
    assert "Dropout helps. [p1_c1]" in result.text and "- Bullet finding [p2_c1]." in result.text


def test_every_sentence_of_the_output_passes_again():
    text = "## A\nOne [p1_c1]. Two [nope_c2]. Three.\n## B\nFour [p3_c7]."
    once = check_citations(text, KNOWN)
    twice = check_citations(once.text, KNOWN)
    assert twice.removed == [] and twice.text == once.text


def _claims():
    return [
        {"claim_id": "a_c1", "paper_id": "a", "text": "A rises.", "paper_title": "Paper A", "year": 2020},
        {"claim_id": "b_c1", "paper_id": "b", "text": "A falls.", "paper_title": "Paper B", "year": 2021},
        {"claim_id": "a_c2", "paper_id": "a", "text": "C holds.", "paper_title": "Paper A", "year": 2020},
    ]


def test_graph_nodes_and_edges():
    graph = build_graph(_claims(), [{"id": 1, "claim_a": "a_c1", "claim_b": "b_c1", "relation": "contradicts"}])
    assert graph.number_of_nodes() == 3 and graph["a_c1"]["b_c1"]["relation"] == "contradicts"


def test_graph_skips_unknown_endpoints():
    graph = build_graph(_claims(), [{"id": 1, "claim_a": "a_c1", "claim_b": "ghost", "relation": "supports"}])
    assert graph.number_of_edges() == 0


def test_cytoscape_export_shape():
    data = to_cytoscape(build_graph(_claims(), [{"id": 7, "claim_a": "a_c1", "claim_b": "b_c1", "relation": None}]))
    assert {node["data"]["id"] for node in data["elements"]["nodes"]} == {"a_c1", "b_c1", "a_c2"}
    edge = data["elements"]["edges"][0]["data"]
    assert edge["pair_id"] == 7 and edge["relation"] == "unclear"
    colours = {node["data"]["id"]: node["data"]["colour"] for node in data["elements"]["nodes"]}
    assert colours["a_c1"] == colours["a_c2"] != colours["b_c1"]
    assert all(colour in PAPER_COLOURS for colour in colours.values())
    assert len(data["papers"]) == 2


def _seed(session) -> Topic:
    session.add_all([Paper(id="a", file_hash="1" * 64, title="A"), Paper(id="b", file_hash="2" * 64, title="B")])
    session.flush()
    for claim_id, paper, text in (("a_c1", "a", "Dropout raises accuracy."), ("b_c1", "b", "Dropout lowers accuracy.")):
        session.add(
            ClaimRow(
                claim_id=claim_id, paper_id=paper, text=text, intervention="dropout", outcome="accuracy", direction="mixed",
                conditions={"dataset": "X"}, hedged=False, section="results", quote=text, quote_verified=True,
            )
        )
    topic = Topic(topic_key="k", label="Dropout", source="auto")
    session.add(topic)
    session.flush()
    session.add_all([ClaimTopic(claim_id="a_c1", topic_id=topic.id, score=0.9), ClaimTopic(claim_id="b_c1", topic_id=topic.id, score=0.8)])
    session.add(ClaimPair(topic_id=topic.id, pair_key="pk", claim_a="a_c1", claim_b="b_c1", relation="contradicts", condition_diff={"explanation": "Same data."}))
    session.commit()
    session.refresh(topic)
    return topic


def test_report_removes_invented_citations_and_is_stored(session):
    topic = _seed(session)
    chain = RunnableLambda(lambda _: "## Where the papers conflict\nThey disagree [a_c1] [b_c1]. Made up [x_c9].")
    report = generate_report(session, topic, chain=chain)
    assert "Made up" not in report.text and set(report.cited_claim_ids) == {"a_c1", "b_c1"}
    assert len(report.removed_sentences) == 1 and report.prompt_version == "write_report.v1"
    assert latest_report(session, topic.id).id == report.id


def test_report_is_cached_until_regenerated(session):
    topic = _seed(session)
    calls = []

    def write(_):
        calls.append(1)
        return "Finding [a_c1]."

    chain = RunnableLambda(write)
    first = generate_report(session, topic, chain=chain)
    assert generate_report(session, topic, chain=chain).id == first.id and len(calls) == 1
    assert generate_report(session, topic, chain=chain, regenerate=True).id != first.id and len(calls) == 2


def test_report_falls_back_to_template_when_nothing_survives(session):
    topic = _seed(session)
    report = generate_report(session, topic, chain=RunnableLambda(lambda _: "Nothing cited here at all."))
    assert report.model == "template" and set(report.cited_claim_ids) == {"a_c1", "b_c1"}
    assert "## Where the papers conflict" in report.text


def test_report_falls_back_when_the_model_fails(session):
    topic = _seed(session)

    def boom(_):
        raise RuntimeError("no model")

    assert generate_report(session, topic, chain=RunnableLambda(boom)).model == "template"


def test_template_with_no_relations_lists_claims():
    claims = [{"claim_id": "a_c1", "text": "A finding."}]
    assert "A finding. [a_c1]" in template_report(claims, [])


CLAIMS = [
    {"claim_id": "a_c1", "text": "Dropout raises accuracy by 2 percent.", "conditions": {"dataset": "CIFAR-10"}},
    {"claim_id": "b_c1", "text": "Dropout lowers accuracy.", "conditions": {"dataset": "ImageNet"}},
    {"claim_id": "c_c1", "text": "Weight decay helps.", "conditions": {}},
]


def _relation(a, b, relation, **extra):
    return {"claim_a": a, "claim_b": b, "relation": relation, "reason": None, "differing_fields": [], **extra}


def test_conflict_written_about_a_different_setting_pair_is_removed():
    """Seen with the real model: 'the papers conflict', with valid citations, on a pair stored as different_setting."""
    relations = [_relation("a_c1", "b_c1", "different_setting", differing_fields=["dataset"])]
    text = "## Where the papers conflict\nThese two disagree [a_c1] [b_c1]."
    grounded = ground_sentences(text, CLAIMS, relations)
    assert not grounded.sentences and "no stored contradicts pair" in grounded.removed[0]["reason"]
    report = compose_report(CLAIMS, relations, grounded)
    assert "Where the papers conflict" not in report and "Where the settings differ" in report


def test_agreement_needs_a_stored_supports_pair_not_two_single_citations():
    relations = [_relation("a_c1", "b_c1", "unclear")]
    text = "## Where the papers agree\nDropout raises accuracy [a_c1].\nDropout lowers accuracy [b_c1]."
    grounded = ground_sentences(text, CLAIMS, relations)
    assert not grounded.sentences and len(grounded.removed) == 2


def test_backed_sentence_stays_and_its_pair_is_not_reported_twice():
    relations = [_relation("a_c1", "b_c1", "contradicts")]
    grounded = ground_sentences("## Where the papers conflict\n- The two disagree [a_c1] [b_c1].", CLAIMS, relations)
    report = compose_report(CLAIMS, relations, grounded)
    assert report == "## Where the papers conflict\nThe two disagree [a_c1] [b_c1]."


def test_repeated_sentences_are_removed():
    relations = [_relation("a_c1", "b_c1", "contradicts")]
    line = "The two disagree [a_c1] [b_c1]."
    grounded = ground_sentences(f"## Where the papers conflict\n{line}\n{line}", CLAIMS, relations)
    assert grounded.sentences["contradicts"] == [line] and grounded.removed[0]["reason"] == "repeated"


def test_number_the_cited_claims_do_not_state_is_removed():
    relations = [_relation("a_c1", "b_c1", "contradicts")]
    text = "## Where the papers conflict\nDropout changed accuracy by 37 percent [a_c1] [b_c1]."
    grounded = ground_sentences(text, CLAIMS, relations)
    assert not grounded.sentences and "37" in grounded.removed[0]["reason"]
    kept = ground_sentences("## Where the papers conflict\nOne gained 2 percent, one fell [a_c1] [b_c1].", CLAIMS, relations)
    assert kept.sentences["contradicts"]


def test_model_text_about_settings_is_replaced_by_stored_conditions():
    """Seen with the real model: 'the metric used is anomaly detection', which no claim stated."""
    relations = [_relation("a_c1", "b_c1", "different_setting", differing_fields=["dataset"])]
    grounded = ground_sentences("## Where the settings differ\nThe metric used is anomaly detection [a_c1] [b_c1].", CLAIMS, relations)
    report = compose_report(CLAIMS, relations, grounded)
    assert "anomaly" not in report
    assert "dataset: CIFAR-10 in the first, ImageNet in the second" in report


def test_relation_the_model_skipped_is_still_reported():
    relations = [_relation("a_c1", "b_c1", "contradicts"), _relation("a_c1", "c_c1", "supports")]
    grounded = ground_sentences("## Where the papers agree\nThey agree [a_c1] [c_c1].", CLAIMS, relations)
    report = compose_report(CLAIMS, relations, grounded)
    assert "They agree" in report and "conflicts with" in report


def test_sentence_outside_the_known_headings_is_removed():
    grounded = ground_sentences("## Something else\nA finding [a_c1].\nAnother [b_c1].", CLAIMS, [])
    assert not grounded.sentences and len(grounded.removed) == 2


def test_unclear_list_is_bounded():
    many = [{"claim_id": f"p{i}_c1", "text": f"Finding {i}.", "conditions": {}} for i in range(12)]
    relations = [_relation(f"p{i}_c1", f"p{i + 1}_c1", "unclear") for i in range(11)]
    assert compose_report(many, relations).count("cannot yet be compared") == 8


def test_model_is_not_called_when_there_is_nothing_for_it_to_phrase(session):
    topic = _seed(session)
    pair = session.scalars(select(ClaimPair)).one()
    pair.relation, pair.condition_diff = "different_setting", {"differing_fields": ["dataset"]}
    session.commit()
    session.refresh(topic)

    def must_not_run(_):
        raise AssertionError("the model should not be called")

    report = generate_report(session, topic, chain=RunnableLambda(must_not_run))
    assert report.model == "template" and "Where the settings differ" in report.text
    assert set(report.cited_claim_ids) == {"a_c1", "b_c1"}


def test_model_never_sees_different_setting_pairs(session):
    topic = _seed(session)
    seen = {}

    def write(inputs):
        seen.update(inputs)
        return "## Where the papers conflict\nThe two disagree [a_c1] [b_c1]."

    report = generate_report(session, topic, chain=RunnableLambda(write))
    assert "different_setting" not in seen["relations"] and report.prompt_version == "write_report.v1"
