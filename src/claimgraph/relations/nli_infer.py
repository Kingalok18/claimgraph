"""NLI cross-encoder. Uses the fine-tuned model if present, else the base model. Scores both directions."""
from __future__ import annotations

import gc
import logging
from dataclasses import dataclass

from ..config import get_settings

log = logging.getLogger(__name__)
LABELS = ("contradiction", "entailment", "neutral")


@dataclass
class NLIResult:
    label: str
    prob: float
    probs: dict[str, float]


class NLIModel:
    def __init__(self, model_path: str | None = None) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        settings = get_settings()
        config = settings.models["nli"]
        self.path = model_path or (
            str(settings.nli_finetuned_path) if settings.nli_finetuned_path.exists() else config["base_model_id"]
        )
        self.device = config.get("device", "cpu")
        self.batch_size = int(config.get("batch_size", 8))
        self.max_length = int(config.get("max_length", 256))
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path).to(self.device).eval()
        # The label order differs between models, so read it from the model's own config.
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self._torch = torch

    def _forward(self, premises: list[str], hypotheses: list[str]):
        torch = self._torch
        outputs = []
        for start in range(0, len(premises), self.batch_size):
            batch = self.tokenizer(
                premises[start : start + self.batch_size],
                hypotheses[start : start + self.batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                outputs.append(torch.softmax(self.model(**batch).logits, dim=-1))
        return torch.cat(outputs, dim=0)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Average of both directions. Entailment is not symmetric; a claim pair should hold both ways."""
        if not pairs:
            return []
        first = [a for a, _ in pairs]
        second = [b for _, b in pairs]
        return self._results((self._forward(first, second) + self._forward(second, first)) / 2)

    def score_one_way(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Premise to hypothesis only, for benchmarks such as SciFact where evidence supports a claim one way."""
        if not pairs:
            return []
        return self._results(self._forward([a for a, _ in pairs], [b for _, b in pairs]))

    def _results(self, scores) -> list[NLIResult]:
        results = []
        for row in scores.tolist():
            probs = {self.id2label[i]: float(p) for i, p in enumerate(row)}
            label = max(probs, key=probs.get)
            results.append(NLIResult(label, probs[label], probs))
        return results


_model: NLIModel | None = None


def get_nli_model() -> NLIModel:
    global _model
    if _model is None:
        _model = NLIModel()
    return _model


def release_nli_model() -> None:
    """Free the model's RAM between pipeline steps."""
    global _model
    _model = None
    gc.collect()
