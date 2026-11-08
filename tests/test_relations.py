import numpy as np
import pytest

from fakes import FakeChatStructured
from claimgraph.claims.schema import Conditions
from claimgraph.relations.compare_chain import (
    ConditionComparison,
    build_compare_chain,
    compare_conditions,
    compare_field,
    rule_compare,
)
from claimgraph.relations.decide import decide
from claimgraph.topics.group import cluster_vectors, generate_pairs, pair_key, topic_key
from claimgraph.topics.label_chain import clean_label, fallback_label, label_topic


@pytest.mark.parametrize(
    "label,prob,match,dir_a,dir_b,expected",
    [
        ("entailment", 0.9, "same", "increase", "increase", "supports"),
        ("entailment", 0.9, "different", "increase", "increase", "supports"),
        ("entailment", 0.6, "unknown", None, None, "supports"),  # threshold is inclusive
        ("contradiction", 0.9, "same", "increase", "decrease", "contradicts"),
        ("contradiction", 0.9, "different", "increase", "decrease", "different_setting"),
        ("contradiction", 0.9, "unknown", "increase", "decrease", "unclear"),
        ("contradiction", 0.6, "same", None, None, "contradicts"),
        ("contradiction", 0.59, "different", "increase", "decrease", "different_setting"),  # falls to the direction rule
        ("contradiction", 0.59, "same", "increase", "decrease", "unclear"),
        ("entailment", 0.59, "same", "increase", "increase", "unclear"),
        ("neutral", 0.9, "different", "increase", "decrease", "different_setting"),
        ("neutral", 0.9, "different", "decrease", "increase", "different_setting"),
        ("neutral", 0.9, "different", "increase", "increase", "unclear"),
        ("neutral", 0.9, "same", "increase", "decrease", "unclear"),
        ("neutral", 0.9, "unknown", "no_effect", "mixed", "unclear"),
    ],
)
def test_decide_every_branch(label, prob, match, dir_a, dir_b, expected):
    assert decide(label, prob, match, dir_a, dir_b) == expected


def test_decide_thresholds_come_from_arguments():
    assert decide("contradiction", 0.7, "same", contradict_th=0.8) == "unclear"
    assert decide("entailment", 0.7, "same", support_th=0.8) == "unclear"


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("CIFAR-10", "cifar 10", "same"),
        ("CIFAR10", "CIFAR-10 dataset", "same"),
        ("CIFAR-10", "ImageNet", "differ"),
        ("CIFAR-10", "CIFAR-100", "uncertain"),
        ("ImageNet", "ImageNet-1k", "uncertain"),
        ("Tampered IC13", "IC13", "uncertain"),
        (None, "ImageNet", "missing"),
        ("the dataset", "ImageNet", "missing"),
    ],
)
def test_field_comparison(a, b, expected):
    assert compare_field(a, b) == expected


def test_rules_decide_same():
    a = Conditions(dataset="CIFAR-10", model="ResNet-50", metric="accuracy")
    b = Conditions(dataset="cifar10", model="ResNet 50", metric="Accuracy")
    assert rule_compare(a, b).match == "same"


def test_rules_decide_different_and_name_the_field():
    result = rule_compare(Conditions(dataset="CIFAR-10", model="ResNet-50"), Conditions(dataset="ImageNet", model="ResNet-50"))
    assert result.match == "different" and result.differing_fields == ["dataset"]


def test_rules_are_inconclusive_without_a_shared_anchor():
    assert rule_compare(Conditions(metric="accuracy"), Conditions(metric="accuracy")).match is None
    assert rule_compare(Conditions(), Conditions()).match is None


def test_rules_send_differing_setting_text_to_the_llm():
    a = Conditions(dataset="X", setting="white box attack")
    b = Conditions(dataset="X", setting="epsilon of 8")
    result = rule_compare(a, b)
    assert result.match is None and "setting" in result.differing_fields


def test_compare_uses_rules_without_calling_the_llm():
    a = Conditions(dataset="CIFAR-10")
    result = compare_conditions("a", a, "b", a, chain="must not be called")
    assert result["decided_by"] == "rules" and result["conditions_match"] == "same"


def test_compare_calls_the_llm_when_inconclusive():
    answer = ConditionComparison(conditions_match="different", differing_fields=["setting", "invented"], explanation="Attacks differ.")
    chain = build_compare_chain(llm=FakeChatStructured(answer))
    result = compare_conditions("a", Conditions(metric="f1"), "b", Conditions(metric="f1"), chain=chain)
    assert result["decided_by"] == "llm" and result["conditions_match"] == "different"
    assert result["differing_fields"] == ["setting"]  # an invented field is discarded


def test_model_explanation_is_not_shown_as_fact():
    """Seen with the real model: the reason named a metric the condition table showed as not stated."""
    from claimgraph.relations.compare_chain import reason_for

    stored = {"decided_by": "llm", "conditions_match": "different", "differing_fields": ["dataset", "metric"], "explanation": "The metric is anomaly detection."}
    assert reason_for(stored) == "The dataset, metric differ."
    assert reason_for({"decided_by": "rules", "explanation": "The dataset differs."}) == "The dataset differs."
    assert reason_for({"decided_by": "llm", "conditions_match": "same", "explanation": "x"}) == "The stated conditions match."
    assert reason_for({"decided_by": "llm", "conditions_match": "unknown", "explanation": "x"}) == "The stated conditions are not enough to compare."
    assert reason_for(None) is None


def test_compare_without_llm_stays_unknown():
    result = compare_conditions("a", Conditions(), "b", Conditions(), use_llm=False)
    assert result["conditions_match"] == "unknown" and result["decided_by"] == "rules"


def test_compare_llm_failure_is_unknown_not_a_crash():
    from langchain_core.runnables import RunnableLambda

    def boom(_):
        raise ValueError("no")

    result = compare_conditions("a", Conditions(), "b", Conditions(), chain=RunnableLambda(boom))
    assert result["conditions_match"] == "unknown" and result["decided_by"] == "error"


def _vector(*values):
    return np.array(values, dtype=float)


def test_clusters_separate_two_subjects():
    vectors = np.vstack(
        [_vector(1, 0.01 * i, 0) for i in range(5)] + [_vector(0, 0.01 * i, 1) for i in range(5)]
    )
    labels = cluster_vectors(vectors)
    assert len(set(labels[:5])) == 1 and len(set(labels[5:])) == 1
    assert labels[0] != labels[5] and -1 not in labels


def test_identical_claims_end_up_in_one_topic():
    """Seen with the real model: HDBSCAN split identical points into arbitrary pairs."""
    same_a, same_b, other = _vector(1, 0, 0), _vector(0.93, 0.37, 0), _vector(0, 0.2, 1)
    vectors = np.vstack([same_a] * 4 + [same_b] * 4 + [other] * 4)
    labels = cluster_vectors(vectors)
    assert len(set(labels[:8])) == 1, labels
    assert len(set(labels[8:])) == 1 and labels[8] != labels[0], labels


def test_merge_threshold_keeps_different_subjects_apart():
    vectors = np.vstack([_vector(1, 0, 0)] * 3 + [_vector(0.7, 0.71, 0)] * 3)
    labels = cluster_vectors(vectors, merge_similarity=0.95)
    assert labels[0] != labels[3]


def test_too_few_points_make_no_cluster():
    assert list(cluster_vectors(np.array([[1.0, 0.0]]))) == [-1]
    assert cluster_vectors(np.zeros((0, 3))).size == 0


def test_outlier_stays_out_when_not_similar():
    vectors = np.vstack([_vector(1, 0.01 * i, 0) for i in range(6)] + [_vector(0, 0, 1)])
    labels = cluster_vectors(vectors, attach_similarity=0.9)
    assert labels[-1] == -1


def test_pairs_only_cross_papers_and_are_unique():
    ids = ["a1", "a2", "b1", "c1"]
    papers = {"a1": "A", "a2": "A", "b1": "B", "c1": "C"}
    vectors = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]], dtype=float)
    pairs = generate_pairs(ids, vectors, papers, neighbours=10)
    assert len(pairs) == len(set(pairs))
    assert ("a1", "a2") not in pairs
    assert all(papers[x] != papers[y] for x, y in pairs)


def test_pairs_limit_is_per_claim_nearest_first():
    ids = ["a1", "b1", "c1", "d1"]
    papers = {claim: claim[0] for claim in ids}
    vectors = np.array([[1, 0], [0.99, 0.01], [0.5, 0.5], [0, 1]], dtype=float)
    pairs = generate_pairs(ids, vectors, papers, neighbours=1)
    assert ("a1", "b1") in pairs and ("a1", "c1") not in pairs


def test_single_paper_makes_no_pairs():
    assert generate_pairs(["a1", "a2"], np.eye(2), {"a1": "A", "a2": "A"}) == []


def test_keys_do_not_depend_on_order():
    assert pair_key("x", "y") == pair_key("y", "x")
    assert topic_key(["a", "b"]) == topic_key(["b", "a"])
    assert topic_key(["a"], "question") != topic_key(["a"])


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('Label: "Dropout effect on accuracy."', "Dropout effect on accuracy"),
        ("**Novel insights into forgery detection**", "into forgery detection"),
        ("one two three four five six seven eight nine ten", "one two three four five six seven eight"),
        ("\n\nAttack strength\nand more lines", "Attack strength"),
    ],
)
def test_labels_are_cleaned(raw, expected):
    assert clean_label(raw) == expected


def test_label_falls_back_when_the_model_fails():
    from langchain_core.runnables import RunnableLambda

    def boom(_):
        raise RuntimeError("no model")

    claims = [{"text": "t", "intervention": "dropout", "outcome": "accuracy"}]
    assert label_topic(claims, chain=RunnableLambda(boom)) == fallback_label(claims) == "dropout: accuracy"
