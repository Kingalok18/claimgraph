"""BioDivergence evaluation (optional dataset). Run: python -m claimgraph.eval.biodivergence_eval

The dataset's licence must be checked by the user first, so it is never downloaded automatically.
Place claim pairs at data/datasets/biodivergence/pairs.jsonl, one JSON object per line, with fields
claim_a, claim_b and relation (supports, contradicts, different_setting or unclear).
"""
from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from .common import classification_metrics, save_result, waiting
from .scifact_nli_eval import read_jsonl

RELATIONS = ["supports", "contradicts", "different_setting", "unclear"]


def nli_to_relation(label: str) -> str:
    """Without condition fields, only the NLI part of the rule applies."""
    return {"entailment": "supports", "contradiction": "contradicts"}.get(label, "unclear")


def main() -> Path:
    from ..relations.nli_infer import NLIModel

    path = get_settings().datasets_dir / "biodivergence" / "pairs.jsonl"
    if not path.exists():
        return waiting("biodivergence", "Dataset not selected at setup; check its licence first. See the README.", "not_selected")
    rows = read_jsonl(path)
    predictions = NLIModel().score_pairs([(row["claim_a"], row["claim_b"]) for row in rows])
    metrics = classification_metrics(
        [row["relation"] for row in rows], [nli_to_relation(item.label) for item in predictions], RELATIONS
    )
    return save_result("biodivergence", {"status": "measured", "note": "NLI only, no condition fields", **metrics})


if __name__ == "__main__":
    print(main())
