"""Build a topic from a user's research question, by similarity search over claims."""
from __future__ import annotations

from ..config import get_settings
from ..embed.vectorstore import search_claims


def claims_for_question(question: str) -> list[tuple[str, float]]:
    """Claims similar to the question: the closest top-k, kept only above the threshold."""
    config = get_settings().pipeline["topics"]
    hits = search_claims(question, k=int(config["question_top_k"]))
    threshold = float(config["question_similarity"])
    return [(claim_id, score) for claim_id, score in hits if score >= threshold]
