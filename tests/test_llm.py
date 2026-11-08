"""The model factory, plus chain tests that run the real language model (mark: llm)."""
import pytest

from claimgraph import llm as llm_module
from claimgraph.config import get_settings
from claimgraph.llm import ChatHFGGUF, check_llm, get_chat_model, gguf_path

needs_model = pytest.mark.skipif(
    gguf_path(
        __import__("claimgraph.config", fromlist=["load_settings"]).load_settings().models["llm"]["hf_model_id"],
        __import__("claimgraph.config", fromlist=["load_settings"]).load_settings().models["llm"]["gguf_file"],
        download=False,
    )
    is None,
    reason="The language model is not downloaded. Run: python scripts/run.py setup",
)


def test_factory_builds_the_configured_model():
    assert isinstance(get_chat_model("main"), ChatHFGGUF)
    assert isinstance(get_chat_model("small"), ChatHFGGUF)


def test_factory_rejects_unknown_input():
    with pytest.raises(ValueError):
        get_chat_model("huge")
    with pytest.raises(ValueError):
        get_chat_model("main", provider="nonsense")


def test_both_kinds_share_one_loaded_model_file():
    """Loading two copies would not fit in this machine's RAM."""
    main, small = get_chat_model("main"), get_chat_model("small")
    assert (main.repo_id, main.filename, main.n_ctx, main.n_threads) == (small.repo_id, small.filename, small.n_ctx, small.n_threads)


def test_structured_grammar_cannot_skip_fields():
    """If fields were optional, the model could answer "{}" and every paper would yield no claims."""
    from claimgraph.claims.schema import ExtractedClaimList

    captured = {}

    class Spy(ChatHFGGUF):
        def bind(self, **kwargs):
            captured.update(kwargs)
            return super().bind(**kwargs)

    Spy(repo_id="r", filename="f").with_structured_output(ExtractedClaimList)
    schema = captured["response_format"]["schema"]
    assert schema["required"] == ["claims"] and schema["properties"]["claims"]["maxItems"] == 5
    claim = schema["$defs"]["ExtractedClaim"]
    assert {"text", "direction", "conditions", "hedged", "quote"} <= set(claim["required"])
    assert set(schema["$defs"]["Conditions"]["required"]) == {"dataset", "model", "metric", "setting"}


def test_check_reports_a_model_not_yet_downloaded(monkeypatch):
    monkeypatch.setattr(llm_module, "gguf_path", lambda *a, **k: None)
    status = check_llm()
    assert status["ok"] is False and status["fatal"] is False and "not downloaded" in status["detail"]


def test_check_reports_ollama_problem_when_chosen(monkeypatch):
    monkeypatch.setitem(get_settings().models["llm"], "provider", "ollama")
    monkeypatch.setitem(get_settings().models["llm"], "model", "x")
    monkeypatch.setattr(get_settings(), "ollama_base_url", "http://127.0.0.1:1")
    status = check_llm(timeout=0.2)
    assert status["ok"] is False and status["fatal"] is True and "not reachable" in status["detail"]


def test_calls_are_logged(session):
    from uuid import uuid4

    from sqlalchemy import select

    from claimgraph.llm import LLMCallLogger
    from claimgraph.models_db import LLMCall

    logger = LLMCallLogger()
    run_id = uuid4()
    logger.on_chat_model_start({}, [], run_id=run_id, metadata={"chain_name": "c", "prompt_version": "v1", "ls_model_name": "m"})
    logger.on_llm_end(None, run_id=run_id)
    row = session.scalar(select(LLMCall))
    assert (row.chain_name, row.prompt_version, row.model, row.ok) == ("c", "v1", "m", True)


TEXT = (
    "3 Results\nAdding dropout increased the test accuracy of ResNet-50 on CIFAR-10 from 92.1 percent to 93.4 percent. "
    "The difference was consistent across all five random seeds."
)


@pytest.mark.llm
@needs_model
def test_real_model_returns_valid_claims():
    from claimgraph.claims.extract_chain import build_extract_chain
    from claimgraph.claims.schema import ExtractedClaimList

    output = build_extract_chain().invoke({"title": "Dropout study", "section": "results", "text": TEXT})
    assert isinstance(output, ExtractedClaimList)
    assert output.claims and output.claims[0].direction == "increase"


@pytest.mark.llm
@needs_model
def test_real_model_compare_chain_returns_valid_object():
    from claimgraph.relations.compare_chain import ConditionComparison, build_compare_chain

    result = build_compare_chain().invoke(
        {
            "text_a": "Dropout raises accuracy under a white-box attack.",
            "conditions_a": '{"setting": "white-box attack"}',
            "text_b": "Dropout lowers accuracy with no attack.",
            "conditions_b": '{"setting": "no attack"}',
        }
    )
    assert isinstance(result, ConditionComparison)


@pytest.mark.llm
@needs_model
def test_real_model_label_and_report_chains_return_text():
    from claimgraph.report.report_chain import build_report_chain
    from claimgraph.topics.label_chain import label_topic

    claims = [{"text": "Dropout raises test accuracy on CIFAR-10.", "intervention": "dropout", "outcome": "accuracy"}]
    label = label_topic(claims)
    assert 0 < len(label.split()) <= 8

    report = build_report_chain().invoke({"topic": "Dropout", "claims": '[{"claim_id": "a_c1", "text": "Dropout raises accuracy."}]', "relations": "[]"})
    assert isinstance(report, str) and report.strip()
