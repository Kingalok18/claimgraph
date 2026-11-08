"""Discover topics by clustering claim embeddings, and generate claim pairs inside a topic."""
from __future__ import annotations

import hashlib

import numpy as np
from sklearn.cluster import HDBSCAN


def _normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _merge_close_clusters(labels: np.ndarray, unit: np.ndarray, merge_similarity: float) -> np.ndarray:
    """Join clusters whose centres nearly coincide.

    With min_cluster_size=2, HDBSCAN splits identical points into arbitrary pairs. Claims with the same
    intervention and outcome would land in separate topics, and pairs are only made inside a topic.
    """
    clusters = [label for label in sorted(set(labels.tolist())) if label != -1]
    if len(clusters) < 2:
        return labels
    centres = _normalise(np.vstack([unit[labels == label].mean(axis=0) for label in clusters]))
    similarity = centres @ centres.T
    parent = list(range(len(clusters)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for first in range(len(clusters)):
        for second in range(first + 1, len(clusters)):
            if similarity[first, second] >= merge_similarity:
                parent[find(second)] = find(first)
    merged = labels.copy()
    for index, label in enumerate(clusters):
        merged[labels == label] = clusters[find(index)]
    return merged


def cluster_vectors(
    vectors: np.ndarray, min_cluster_size: int = 2, attach_similarity: float = 0.6, merge_similarity: float = 0.85
) -> np.ndarray:
    """Cluster labels for each vector; -1 means no topic.

    Euclidean distance on unit vectors orders points the same way as cosine distance.
    Leftover points join the closest cluster when they are similar enough to it.
    """
    count = len(vectors)
    if count < max(2, min_cluster_size):
        return np.full(count, -1)
    unit = _normalise(np.asarray(vectors, dtype=float))
    # A small library may be one subject. On a large one, a single cluster would hide all structure.
    model = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", allow_single_cluster=count < 20, copy=True)
    labels = _merge_close_clusters(model.fit_predict(unit), unit, merge_similarity)

    clusters = [label for label in sorted(set(labels)) if label != -1]
    if clusters:
        centroids = _normalise(np.vstack([unit[labels == label].mean(axis=0) for label in clusters]))
        for index in np.where(labels == -1)[0]:
            similarity = centroids @ unit[index]
            best = int(np.argmax(similarity))
            if similarity[best] >= attach_similarity:
                labels[index] = clusters[best]
    return labels


def topic_key(claim_ids: list[str], extra: str = "") -> str:
    return hashlib.sha1(("|".join(sorted(claim_ids)) + extra).encode()).hexdigest()


def pair_key(claim_a: str, claim_b: str) -> str:
    first, second = sorted((claim_a, claim_b))
    return hashlib.sha1(f"{first}|{second}".encode()).hexdigest()[:32]


def generate_pairs(
    claim_ids: list[str], vectors: np.ndarray, paper_of: dict[str, str], neighbours: int = 10
) -> list[tuple[str, str]]:
    """Pair each claim with its nearest claims from OTHER papers. Each pair appears once, sorted."""
    if len(claim_ids) < 2:
        return []
    unit = _normalise(np.asarray(vectors, dtype=float))
    similarity = unit @ unit.T
    papers = np.array([paper_of[claim_id] for claim_id in claim_ids])
    pairs: set[tuple[str, str]] = set()
    for index, claim_id in enumerate(claim_ids):
        others = np.where(papers != papers[index])[0]
        if others.size == 0:
            continue
        nearest = others[np.argsort(-similarity[index][others], kind="stable")][:neighbours]
        for other in nearest:
            pairs.add(tuple(sorted((claim_id, claim_ids[int(other)]))))
    return sorted(pairs)
