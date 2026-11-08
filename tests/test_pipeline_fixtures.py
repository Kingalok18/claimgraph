"""End to end with the REAL models on the planted fixtures, compared with expected.json."""
import pytest

from claimgraph.config import load_settings
from claimgraph.llm import gguf_path

_config = load_settings().models["llm"]

pytestmark = [
    pytest.mark.llm,
    pytest.mark.models,
    pytest.mark.skipif(
        gguf_path(_config["hf_model_id"], _config["gguf_file"], download=False) is None,
        reason="The language model is not downloaded. Run: python scripts/run.py setup",
    ),
]


@pytest.fixture(scope="module")
def outcome():
    from claimgraph.eval.fixture_eval import run_fixture_pipeline
    from claimgraph.llm import ChatHFGGUF
    from claimgraph.relations.nli_infer import NLIModel

    calls = {"llm": 0, "nli": 0}

    def counted(function, key):
        def wrapper(*args, **kwargs):
            calls[key] += 1
            return function(*args, **kwargs)

        return wrapper

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ChatHFGGUF, "_generate", counted(ChatHFGGUF._generate, "llm"))
        patch.setattr(NLIModel, "score_pairs", counted(NLIModel.score_pairs, "nli"))
        result = run_fixture_pipeline()
    return {**result, "real_model_calls": calls}


def test_the_real_models_did_the_work(outcome):
    assert outcome["real_model_calls"]["llm"] > 0, outcome["real_model_calls"]
    assert outcome["real_model_calls"]["nli"] > 0, outcome["real_model_calls"]


def test_every_planted_claim_is_found(outcome):
    assert outcome["claims_found"] == outcome["claims_planted"], outcome["planted"]


def test_all_stored_quotes_are_verified(outcome):
    assert outcome["claims_stored"] >= outcome["claims_planted"]
    assert outcome["all_quotes_verified"] is True


@pytest.mark.parametrize("relation", ["supports", "contradicts", "different_setting"])
def test_planted_relation_comes_out(outcome, relation):
    rows = [row for row in outcome["relations"] if row["expected"] == relation]
    assert rows and all(row["ok"] for row in rows), rows


def test_report_puts_each_planted_relation_under_its_heading(outcome):
    report = outcome["report"]
    assert report["relations_in_right_section"] == report["relations_checked"], report
