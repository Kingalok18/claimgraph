"""Cytoscape.js elements for the graph screen. Papers get muted colours from a fixed palette."""
from __future__ import annotations

import networkx as nx

PAPER_COLOURS = ["#0F5F5F", "#7A5C3E", "#5A6B8C", "#8A5A7A", "#4F7A5A", "#9A6B3A", "#5E5E8C", "#7E6A4A", "#3F6F7F", "#8C5E5E"]


def paper_colour(index: int) -> str:
    return PAPER_COLOURS[index % len(PAPER_COLOURS)]


def _short(text: str, limit: int = 60) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def to_cytoscape(graph: nx.Graph) -> dict:
    papers: dict[str, int] = {}
    nodes = []
    for claim_id, data in graph.nodes(data=True):
        paper_id = data.get("paper_id", "")
        index = papers.setdefault(paper_id, len(papers))
        nodes.append(
            {
                "data": {
                    "id": claim_id,
                    "label": _short(data.get("text", claim_id)),
                    "text": data.get("text", ""),
                    "paper_id": paper_id,
                    "paper_title": data.get("paper_title", ""),
                    "year": data.get("year"),
                    "colour": paper_colour(index),
                }
            }
        )
    edges = []
    for source, target, data in graph.edges(data=True):
        edges.append(
            {
                "data": {
                    "id": f"e{data.get('id')}",
                    "pair_id": data.get("id"),
                    "source": source,
                    "target": target,
                    "relation": data.get("relation") or "unclear",
                }
            }
        )
    legend = [
        {
            "paper_id": paper_id,
            "colour": paper_colour(index),
            "title": next((d.get("paper_title", "") for _, d in graph.nodes(data=True) if d.get("paper_id") == paper_id), ""),
        }
        for paper_id, index in papers.items()
    ]
    return {"elements": {"nodes": nodes, "edges": edges}, "papers": legend}
