"""Shared helpers. Metrics exist only as JSON written here, each with its run date."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import get_settings


def save_result(name: str, payload: dict, folder: Path | None = None) -> Path:
    folder = folder or get_settings().results_dir
    folder.mkdir(parents=True, exist_ok=True)
    data = {"name": name, "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), **payload}
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def waiting(name: str, reason: str, status: str = "waiting_for_labels") -> Path:
    """Record an evaluation that cannot run yet. It carries no numbers."""
    return save_result(name, {"status": status, "reason": reason})


def classification_metrics(gold: list[str], predicted: list[str], labels: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(gold, predicted, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(gold, predicted, labels=labels, average="macro", zero_division=0)
    return {
        "n": len(gold),
        "accuracy": round(float(accuracy_score(gold, predicted)), 4),
        "macro_f1": round(float(macro[2]), 4),
        "per_class": {
            label: {
                "precision": round(float(precision[index]), 4),
                "recall": round(float(recall[index]), 4),
                "f1": round(float(f1[index]), 4),
                "support": int(support[index]),
            }
            for index, label in enumerate(labels)
        },
    }
