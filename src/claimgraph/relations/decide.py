"""The core research rule (spec Section 8.6): a contradiction needs a conflict AND comparable conditions."""
from __future__ import annotations

OPPOSITE = {("increase", "decrease"), ("decrease", "increase")}
RELATIONS = ("supports", "contradicts", "different_setting", "unclear")


def decide(
    nli_label: str,
    nli_prob: float,
    conditions_match: str,
    direction_a: str | None = None,
    direction_b: str | None = None,
    support_th: float = 0.6,
    contradict_th: float = 0.6,
) -> str:
    """Return the relation for a pair. conditions_match is 'same', 'different' or 'unknown'."""
    if nli_label == "entailment" and nli_prob >= support_th:
        return "supports"
    if nli_label == "contradiction" and nli_prob >= contradict_th:
        if conditions_match == "same":
            return "contradicts"
        if conditions_match == "different":
            return "different_setting"
        return "unclear"
    if (direction_a, direction_b) in OPPOSITE and conditions_match == "different":
        return "different_setting"
    return "unclear"
