"""How well does the NLI model label SciFact evidence? Run: python -m claimgraph.eval.scifact_nli_eval"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..config import get_settings
from .common import classification_metrics, save_result, waiting

LABEL_MAP = {"SUPPORT": "entailment", "CONTRADICT": "contradiction"}
LABELS = ["entailment", "contradiction", "neutral"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_examples(claims: list[dict], corpus: list[dict]) -> list[dict]:
    """Evidence sentences support or refute a claim. A cited abstract with no evidence is neutral."""
    abstracts = {str(doc["doc_id"]): doc["abstract"] for doc in corpus}
    examples: list[dict] = []
    for claim in claims:
        evidence = claim.get("evidence") or {}
        for doc_id, groups in evidence.items():
            sentences = abstracts.get(str(doc_id))
            if not sentences:
                continue
            for group in groups:
                label = LABEL_MAP.get(group.get("label"))
                if label:
                    premise = " ".join(sentences[index] for index in group["sentences"] if index < len(sentences))
                    examples.append({"premise": premise, "hypothesis": claim["claim"], "label": label})
        if not evidence:
            for doc_id in claim.get("cited_doc_ids", [])[:1]:
                sentences = abstracts.get(str(doc_id))
                if sentences:
                    examples.append({"premise": " ".join(sentences[:3]), "hypothesis": claim["claim"], "label": "neutral"})
    return examples


def main(limit: int | None = None) -> Path:
    from ..relations.nli_infer import NLIModel

    folder = get_settings().datasets_dir / "scifact"
    claims_path, corpus_path = folder / "claims_dev.jsonl", folder / "corpus.jsonl"
    if not claims_path.exists() or not corpus_path.exists():
        return waiting("scifact_nli", "SciFact is not downloaded. Run: python scripts/download_data.py", "waiting_for_data")

    examples = build_examples(read_jsonl(claims_path), read_jsonl(corpus_path))
    if limit:
        examples = examples[:limit]
    model = NLIModel()
    pairs = [(item["premise"], item["hypothesis"]) for item in examples]
    gold = [item["label"] for item in examples]
    # The pipeline averages both directions, because a claim pair should hold both ways. SciFact asks a
    # one-way question (does this evidence support this claim?), so both readings are reported.
    both = classification_metrics(gold, [item.label for item in model.score_pairs(pairs)], LABELS)
    one_way = classification_metrics(gold, [item.label for item in model.score_one_way(pairs)], LABELS)
    return save_result(
        "scifact_nli",
        {"status": "measured", "split": "dev", "model": model.path, "both_directions": True, **both, "one_way": one_way},
    )


if __name__ == "__main__":
    print(main(int(sys.argv[1]) if len(sys.argv) > 1 else None))
