"""Database engine and sessions. Works with MySQL (this install) and SQLite (tests, fallback)."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models_db import Base

_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(connection, _record):  # pragma: no cover - trivial
            cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        return engine
    # pool_pre_ping survives MySQL closing idle connections during long model steps.
    return create_engine(url, pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=5)


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        settings.ensure_dirs()
        _engine = make_engine(settings.database_url)
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def reset_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None


def init_db() -> None:
    Base.metadata.create_all(get_engine())


def new_session() -> Session:
    get_engine()
    assert _session_factory is not None
    return _session_factory()


@contextmanager
def session_scope() -> Iterator[Session]:
    session = new_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database() -> dict:
    """Report whether the configured database answers. Never raises and never shows the password."""
    url = get_settings().database_url
    kind = "mysql" if url.startswith("mysql") else "sqlite" if url.startswith("sqlite") else "other"
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"ok": True, "kind": kind, "detail": "connected"}
    except Exception as exc:  # noqa: BLE001 - shown on the setup screen
        return {"ok": False, "kind": kind, "detail": _short_db_error(exc)}


def _short_db_error(exc: Exception) -> str:
    message = str(getattr(exc, "orig", exc))
    if "Access denied" in message:
        return "MySQL refused the login. Run data/setup_mysql.sql once (see README)."
    if "Unknown database" in message:
        return "The claimgraph database does not exist yet. Run data/setup_mysql.sql once (see README)."
    if "Can't connect" in message or "Connection refused" in message:
        return "MySQL is not reachable. Start the MySQL80 service."
    return message[:200]
