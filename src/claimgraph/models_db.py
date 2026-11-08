"""SQLAlchemy tables (spec Section 7.2).

Columns beyond the spec, each for a reason:
- papers.page_offsets, parse_error, extract_status, chunks_done, missing: resume long runs, map quotes to pages.
- sections.heading: the raw heading for the outline (name holds the canonical section).
- topics.topic_key: keeps a topic, its label and report when its claims have not changed.
- claim_pairs.pair_key: identifies a pair across re-analysis so user labels survive.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# MySQL TEXT stops at 64 KB, and a long section can pass that.
LongText = Text().with_variant(LONGTEXT(), "mysql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    file_path: Mapped[str | None] = mapped_column(String(1024))
    file_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str | None] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    venue: Mapped[str | None] = mapped_column(String(512))
    doi: Mapped[str | None] = mapped_column(String(255))
    abstract: Mapped[str | None] = mapped_column(LongText)
    source: Mapped[str] = mapped_column(String(16), default="user")
    parse_status: Mapped[str] = mapped_column(String(16), default="pending")
    parse_error: Mapped[str | None] = mapped_column(Text)
    n_pages: Mapped[int | None] = mapped_column(Integer)
    page_offsets: Mapped[list | None] = mapped_column(JSON)
    extract_status: Mapped[str] = mapped_column(String(16), default="pending")
    chunks_done: Mapped[int] = mapped_column(Integer, default=0)
    chunks_total: Mapped[int] = mapped_column(Integer, default=0)
    claims_dropped: Mapped[int] = mapped_column(Integer, default=0)
    missing: Mapped[bool] = mapped_column(Boolean, default=False)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    sections: Mapped[list["Section"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", order_by="Section.char_start"
    )
    claims: Mapped[list["ClaimRow"]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    heading: Mapped[str | None] = mapped_column(String(512))
    text: Mapped[str] = mapped_column(LongText)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    page_start: Mapped[int | None] = mapped_column(Integer)

    paper: Mapped[Paper] = relationship(back_populates="sections")


class ClaimRow(Base):
    __tablename__ = "claims"

    claim_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(Text)
    intervention: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text)
    direction: Mapped[str] = mapped_column(String(16))
    conditions: Mapped[dict] = mapped_column(JSON)
    hedged: Mapped[bool] = mapped_column(Boolean, default=False)
    section: Mapped[str] = mapped_column(String(64))
    page: Mapped[int | None] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)
    quote_span_start: Mapped[int | None] = mapped_column(Integer)
    quote_span_end: Mapped[int | None] = mapped_column(Integer)
    quote_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    paper: Mapped[Paper] = relationship(back_populates="claims")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_key: Mapped[str] = mapped_column(String(64), index=True)
    label: Mapped[str] = mapped_column(String(255))
    source: Mapped[str] = mapped_column(String(16), default="auto")
    question_text: Mapped[str | None] = mapped_column(Text)

    claim_links: Mapped[list["ClaimTopic"]] = relationship(cascade="all, delete-orphan", back_populates="topic")
    pairs: Mapped[list["ClaimPair"]] = relationship(cascade="all, delete-orphan", back_populates="topic")
    reports: Mapped[list["Report"]] = relationship(cascade="all, delete-orphan", back_populates="topic")


class ClaimTopic(Base):
    __tablename__ = "claim_topics"

    claim_id: Mapped[str] = mapped_column(ForeignKey("claims.claim_id", ondelete="CASCADE"), primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), primary_key=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)

    topic: Mapped[Topic] = relationship(back_populates="claim_links")


class ClaimPair(Base):
    __tablename__ = "claim_pairs"
    __table_args__ = (Index("ix_pairs_topic_key", "topic_id", "pair_key", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pair_key: Mapped[str] = mapped_column(String(40), index=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), index=True)
    claim_a: Mapped[str] = mapped_column(ForeignKey("claims.claim_id", ondelete="CASCADE"))
    claim_b: Mapped[str] = mapped_column(ForeignKey("claims.claim_id", ondelete="CASCADE"))
    nli_label: Mapped[str | None] = mapped_column(String(16))
    nli_prob: Mapped[float | None] = mapped_column(Float)
    conditions_match: Mapped[str | None] = mapped_column(String(16))
    condition_diff: Mapped[dict | None] = mapped_column(JSON)
    relation: Mapped[str | None] = mapped_column(String(24), index=True)

    topic: Mapped[Topic] = relationship(back_populates="pairs")


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(LongText)
    cited_claim_ids: Mapped[list] = mapped_column(JSON)
    removed_sentences: Mapped[list | None] = mapped_column(JSON)
    prompt_version: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    topic: Mapped[Topic] = relationship(back_populates="reports")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    progress: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class Label(Base):
    __tablename__ = "labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # extraction | pair
    target_id: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class LLMCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chain_name: Mapped[str | None] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(128))
    ms: Mapped[int | None] = mapped_column(Integer)
    ok: Mapped[bool] = mapped_column(Boolean, default=True)
