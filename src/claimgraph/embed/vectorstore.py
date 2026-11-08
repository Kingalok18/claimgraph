"""Qdrant claim index: a server if QDRANT_URL is set, otherwise local mode inside data/qdrant."""
from __future__ import annotations

import threading
import uuid

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointIdsList, VectorParams

from ..config import get_settings
from .embeddings import get_embeddings

COLLECTION = "claims"
_client: QdrantClient | None = None
_lock = threading.Lock()


def get_client() -> QdrantClient:
    global _client
    with _lock:
        if _client is None:
            settings = get_settings()
            if settings.qdrant_url:
                _client = QdrantClient(url=settings.qdrant_url)
            else:
                settings.qdrant_path.mkdir(parents=True, exist_ok=True)
                # Local mode allows one client per folder, hence the shared instance.
                _client = QdrantClient(path=str(settings.qdrant_path))
    return _client


def close_client() -> None:
    global _client
    with _lock:
        if _client is not None:
            _client.close()
        _client = None


def point_id(claim_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, claim_id))


def claim_embed_text(intervention: str, outcome: str) -> str:
    return f"{intervention} -> {outcome}"


def _has_collection() -> bool:
    return get_client().collection_exists(COLLECTION)


def _store() -> QdrantVectorStore:
    embeddings = get_embeddings()
    client = get_client()
    if not client.collection_exists(COLLECTION):
        size = len(embeddings.embed_query("dimension probe"))
        client.create_collection(COLLECTION, vectors_config=VectorParams(size=size, distance=Distance.COSINE))
    return QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)


def indexed_ids(claim_ids: list[str]) -> set[str]:
    if not claim_ids or not _has_collection():
        return set()
    found = get_client().retrieve(
        COLLECTION, ids=[point_id(c) for c in claim_ids], with_payload=True, with_vectors=False
    )
    return {point.payload["metadata"]["claim_id"] for point in found}


def index_claims(claims: list[dict]) -> int:
    """Index claims not yet present. Each dict has claim_id, paper_id, intervention, outcome."""
    done = indexed_ids([claim["claim_id"] for claim in claims])
    new = [claim for claim in claims if claim["claim_id"] not in done]
    if not new:
        return 0
    _store().add_texts(
        texts=[claim_embed_text(claim["intervention"], claim["outcome"]) for claim in new],
        metadatas=[{"claim_id": claim["claim_id"], "paper_id": claim["paper_id"]} for claim in new],
        ids=[point_id(claim["claim_id"]) for claim in new],
        batch_size=32,
    )
    return len(new)


def remove_claims(claim_ids: list[str]) -> None:
    """Drop claims from the index, for example when their paper is removed."""
    if not claim_ids or not _has_collection():
        return
    get_client().delete(COLLECTION, points_selector=PointIdsList(points=[point_id(c) for c in claim_ids]))


def get_vectors(claim_ids: list[str]) -> dict[str, list[float]]:
    if not claim_ids or not _has_collection():
        return {}
    points = get_client().retrieve(
        COLLECTION, ids=[point_id(c) for c in claim_ids], with_payload=True, with_vectors=True
    )
    return {point.payload["metadata"]["claim_id"]: list(point.vector) for point in points}


def search_claims(query: str, k: int) -> list[tuple[str, float]]:
    if not _has_collection():
        return []
    hits = _store().similarity_search_with_score(query, k=k)
    return [(document.metadata["claim_id"], float(score)) for document, score in hits]
