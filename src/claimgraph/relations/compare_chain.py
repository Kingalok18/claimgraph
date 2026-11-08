"""Are two claims' study conditions comparable? Field rules first, the LLM only if they are inconclusive."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..claims.extract_chain import load_prompt
from ..claims.schema import Conditions
from ..llm import get_chat_model, structured

log = logging.getLogger(__name__)
PROMPT_VERSION = "compare_conditions.v1"
FIELDS = ("dataset", "model", "metric", "setting")
_STOP = {"the", "a", "an", "dataset", "datasets", "data", "set", "benchmark", "on", "of", "model", "method", "score", "test"}


class ConditionComparison(BaseModel):
    conditions_match: Literal["same", "different", "unknown"]
    differing_fields: list[str] = Field(default_factory=list)
    explanation: str = ""


@dataclass
class RuleResult:
    match: str | None  # None = inconclusive, ask the LLM
    differing_fields: list[str] = field(default_factory=list)
    field_status: dict[str, str] = field(default_factory=dict)


def normalise(value: str) -> tuple[str, set[str]]:
    """Compact form and word set. 'CIFAR-10' and 'cifar 10' become 'cifar10'."""
    lowered = re.sub(r"(?<=[a-z0-9])[-_ ](?=\d)", "", value.lower())
    tokens = [token for token in re.findall(r"[a-z0-9]+", lowered) if token not in _STOP]
    return "".join(tokens), set(tokens)


def compare_field(a: str | None, b: str | None) -> str:
    """Returns same, differ, uncertain or missing."""
    if not a or not b:
        return "missing"
    compact_a, tokens_a = normalise(a)
    compact_b, tokens_b = normalise(b)
    if not compact_a or not compact_b:
        return "missing"
    if compact_a == compact_b:
        return "same"
    # Overlap or a shared prefix might be one thing (ImageNet, ImageNet-1k) or two (CIFAR-10, CIFAR-100).
    if tokens_a & tokens_b or compact_a.startswith(compact_b) or compact_b.startswith(compact_a):
        return "uncertain"
    return "differ"


def rule_compare(a: Conditions, b: Conditions) -> RuleResult:
    status = {name: compare_field(getattr(a, name), getattr(b, name)) for name in FIELDS}
    differing = [name for name, value in status.items() if value == "differ"]

    # Dataset, model and metric are names, so a clean mismatch is decisive.
    if any(status[name] == "differ" for name in ("dataset", "model", "metric")):
        return RuleResult("different", differing, status)
    anchored = status["dataset"] == "same" or status["model"] == "same"
    open_questions = any(value in ("uncertain", "differ") for value in status.values())
    if anchored and not open_questions:
        return RuleResult("same", [], status)
    # A "setting" is free text, so a mismatch in its wording needs reading.
    return RuleResult(None, differing, status)


def build_compare_chain(llm=None):
    model = llm if llm is not None else get_chat_model("main")
    chain = load_prompt(PROMPT_VERSION) | structured(model, ConditionComparison)
    return chain.with_retry(stop_after_attempt=2).with_config(
        metadata={"chain_name": "compare_conditions", "prompt_version": PROMPT_VERSION}
    )


def plain_reason(match: str | None, differing_fields: list[str] | None) -> str:
    """One line from the stored comparison, so that it cannot say something the condition table does not."""
    fields = differing_fields or []
    if match == "same":
        return "The stated conditions match."
    if match == "different" and fields:
        return f"The {', '.join(fields)} {'differs' if len(fields) == 1 else 'differ'}."
    if match == "different":
        return "The study conditions differ."
    return "The stated conditions are not enough to compare."


def reason_for(condition_diff: dict | None) -> str | None:
    """The line to show. A model's free-text explanation is kept for the record but is not shown as fact."""
    diff = condition_diff or {}
    if diff.get("decided_by") == "llm":
        return plain_reason(diff.get("conditions_match"), diff.get("differing_fields"))
    return diff.get("explanation")


def compare_conditions(
    text_a: str, conditions_a: Conditions, text_b: str, conditions_b: Conditions, chain=None, use_llm: bool = True
) -> dict:
    """Return conditions_match, differing_fields, field_status, explanation and decided_by."""
    rule = rule_compare(conditions_a, conditions_b)
    if rule.match is not None:
        return {
            "conditions_match": rule.match,
            "differing_fields": rule.differing_fields,
            "field_status": rule.field_status,
            "explanation": plain_reason(rule.match, rule.differing_fields),
            "decided_by": "rules",
        }
    if not use_llm:
        return {
            "conditions_match": "unknown",
            "differing_fields": rule.differing_fields,
            "field_status": rule.field_status,
            "explanation": "The stated conditions are not enough to compare.",
            "decided_by": "rules",
        }
    try:
        chain = chain if chain is not None else build_compare_chain()
        result = chain.invoke(
            {
                "text_a": text_a,
                "conditions_a": json.dumps(conditions_a.model_dump()),
                "text_b": text_b,
                "conditions_b": json.dumps(conditions_b.model_dump()),
            }
        )
        return {
            "conditions_match": result.conditions_match,
            "differing_fields": [name for name in result.differing_fields if name in FIELDS],
            "field_status": rule.field_status,
            "explanation": result.explanation,
            "decided_by": "llm",
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("Condition comparison failed: %s", exc)
        return {
            "conditions_match": "unknown",
            "differing_fields": rule.differing_fields,
            "field_status": rule.field_status,
            "explanation": "The comparison could not be completed.",
            "decided_by": "error",
        }
