"""Claim data model (spec Section 7.1).

ExtractedClaim is what the LLM fills in. Fields the spec marks "set by code" are left out of it,
so the model cannot invent claim ids, pages or verification flags.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Direction = Literal["increase", "decrease", "no_effect", "mixed"]


class Conditions(BaseModel):
    dataset: str | None = None
    model: str | None = None  # or population, for biomedical papers
    metric: str | None = None
    setting: str | None = None

    @field_validator("dataset", "model", "metric", "setting", mode="before")
    @classmethod
    def _blank_to_none(cls, value):
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "n/a", "na", "unknown", "not specified"):
            return None
        return value


class ExtractedClaim(BaseModel):
    # The quote comes first: the model copies from the text before it starts to summarise.
    quote: str = Field(description="Exact sentence copied from the input text")
    text: str = Field(description="The finding as one plain sentence")
    intervention: str = Field(description="What was changed, applied or compared")
    outcome: str = Field(description="What was measured")
    direction: Direction
    conditions: Conditions = Field(default_factory=Conditions)
    hedged: bool = False


class ExtractedClaimList(BaseModel):
    # The limit also bounds the reply, so that a long answer is not cut off into invalid JSON.
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=5)


class Claim(BaseModel):
    claim_id: str  # f"{paper_id}_c{n}"  (set by code)
    paper_id: str  # set by code
    text: str
    intervention: str
    outcome: str
    direction: Direction
    conditions: Conditions
    hedged: bool
    section: str  # set by code
    page: int | None = None  # set by code
    quote: str  # exact sentence from the paper
    quote_span: tuple[int, int] | None = None  # set by code
    quote_verified: bool = False  # set by code


class ClaimList(BaseModel):
    claims: list[Claim]
