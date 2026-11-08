"""Embedding model, loaded on demand and released afterwards to keep RAM free."""
from __future__ import annotations

import gc

from ..config import get_settings

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        config = get_settings().models["embeddings"]
        _embeddings = HuggingFaceEmbeddings(
            model_name=config["model_id"],
            model_kwargs={"device": config.get("device", "cpu")},
            encode_kwargs={"normalize_embeddings": True, "batch_size": int(config.get("batch_size", 8))},
        )
    return _embeddings


def set_embeddings(embeddings) -> None:
    """Replace the model. Tests use this."""
    global _embeddings
    _embeddings = embeddings


def release_embeddings() -> None:
    global _embeddings
    _embeddings = None
    gc.collect()
