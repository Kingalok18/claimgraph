"""Small stand-ins for the models. They test plumbing only; real-model tests carry the llm/models marks."""
from __future__ import annotations

import hashlib
import re

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from claimgraph.claims.schema import Conditions, ExtractedClaim, ExtractedClaimList
from claimgraph.relations.nli_infer import NLIResult


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashing: texts sharing words get similar unit vectors."""

    size = 64

    def _vector(self, text: str) -> list[float]:
        vector = np.zeros(self.size)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.size] += 1.0
        norm = np.linalg.norm(vector)
        return (vector / norm if norm else vector + 1 / np.sqrt(self.size)).tolist()

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


class FakeNLI:
    """Opposite verbs contradict, matching verbs entail. Enough for the decision plumbing."""

    batch_size = 8

    def score_pairs(self, pairs):
        output = []
        for a, b in pairs:
            up_a, up_b = "increased" in a or "increases" in a, "increased" in b or "increases" in b
            label = "entailment" if up_a == up_b else "contradiction"
            output.append(NLIResult(label, 0.9, {"entailment": 0.05, "contradiction": 0.05, "neutral": 0.05, label: 0.9}))
        return output


def planted_claim_from(text: str) -> list[ExtractedClaim]:
    """Read the planted fixture sentence out of a chunk, as a good model would."""
    match = re.search(r"Adding dropout (increased|decreased)[^.]*?on ([A-Za-z0-9-]+) from [\d.]+ percent to [\d.]+ percent\.", text)
    if not match:
        return []
    verb, dataset = match.group(1), match.group(2)
    return [
        ExtractedClaim(
            text=f"Adding dropout {verb} the test accuracy of ResNet-50 on {dataset}.",
            intervention="dropout",
            outcome="test accuracy",
            direction="increase" if verb == "increased" else "decrease",
            conditions=Conditions(dataset=dataset, model="ResNet-50", metric="test accuracy"),
            hedged=False,
            quote=match.group(0),
        )
    ]


def fake_extract_chain(extra_bad_quote: bool = False):
    """Extraction chain stand-in. With extra_bad_quote it also invents a claim whose quote is not in the text."""

    def run(inputs: dict) -> ExtractedClaimList:
        claims = planted_claim_from(inputs["text"])
        if extra_bad_quote and claims:
            claims.append(
                ExtractedClaim(
                    text="An invented finding.",
                    intervention="nothing",
                    outcome="nothing",
                    direction="mixed",
                    quote="This sentence does not appear anywhere in the paper at all, not even nearly.",
                )
            )
        return ExtractedClaimList(claims=claims)

    return RunnableLambda(run)


class FakeChatStructured:
    """Fake model for testing chain construction: returns a fixed object for any prompt."""

    def __init__(self, value):
        self.value = value

    def with_structured_output(self, _schema, **_kwargs):
        return RunnableLambda(lambda _prompt: self.value)
