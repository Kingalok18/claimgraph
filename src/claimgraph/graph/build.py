"""The claim graph of a topic: claims are nodes, relations are edges."""
from __future__ import annotations

import networkx as nx


def build_graph(claims: list[dict], pairs: list[dict]) -> nx.Graph:
    """claims need claim_id; pairs need id, claim_a, claim_b, relation. Unknown endpoints are skipped."""
    graph = nx.Graph()
    for claim in claims:
        graph.add_node(claim["claim_id"], **claim)
    for pair in pairs:
        if pair["claim_a"] in graph and pair["claim_b"] in graph:
            graph.add_edge(pair["claim_a"], pair["claim_b"], **pair)
    return graph
