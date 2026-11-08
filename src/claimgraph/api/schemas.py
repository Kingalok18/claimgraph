"""Request bodies for the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1, description="Analyse only this many new papers")
    paper_ids: list[str] | None = None


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class PairLabelRequest(BaseModel):
    relation: Literal["supports", "contradicts", "different_setting", "unclear"]
    note: str | None = Field(default=None, max_length=2000)


class ClaimLabelRequest(BaseModel):
    claim_id: str
    verdict: Literal["correct", "wrong", "not_a_finding"]
    note: str | None = Field(default=None, max_length=2000)


class OnlineSearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
