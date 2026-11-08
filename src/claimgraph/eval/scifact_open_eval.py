"""SciFact-Open evaluation (optional dataset). Run: python -m claimgraph.eval.scifact_open_eval

SciFact-Open uses the same claim format as SciFact over a far larger corpus. Embedding that corpus
needs a GPU, so this evaluates NLI on its annotated evidence only. It runs if the data was downloaded
with: python scripts/download_data.py --scifact-open
"""
from __future__ import annotations

from pathlib import Path

from ..config import get_settings
from .common import classification_metrics, save_result, waiting
from .scifact_nli_eval import LABELS, build_examples, read_jsonl


def main(limit: int | None = None) -> Path:
    from ..relations.nli_infer import NLIModel

    folder = get_settings().datasets_dir / "scifact_open"
    claims_path, corpus_path = folder / "claims.jsonl", folder / "corpus.jsonl"
    if not claims_path.exists() or not corpus_path.exists():
        return waiting("scifact_open", "Dataset not selected at setup. Optional step: see the README.", "not_selected")

    examples = build_examples(read_jsonl(claims_path), read_jsonl(corpus_path))
    if limit:
        examples = examples[:limit]
    model = NLIModel()
    predictions = model.score_pairs([(item["premise"], item["hypothesis"]) for item in examples])
    metrics = classification_metrics([item["label"] for item in examples], [item.label for item in predictions], LABELS)
    return save_result("scifact_open", {"status": "measured", "model": model.path, **metrics})


if __name__ == "__main__":
    print(main())
