"""Database engine factory and session management."""

from __future__ import annotations

import logging
import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import OperationalError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from abevalflow.db.models import Base

logger = logging.getLogger(__name__)

_DEFAULT_URL = "sqlite:///abevalflow.db"


def get_engine(url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine.

    Args:
        url: Database URL. Falls back to ``DATABASE_URL`` env var,
             then to a local SQLite file.
    """
    db_url = url or os.environ.get("DATABASE_URL", _DEFAULT_URL)

    connect_args: dict = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(db_url, connect_args=connect_args)

    host = engine.url.host or engine.url.database or "in-memory"
    logger.info("Database engine created: dialect=%s host=%s", engine.url.get_backend_name(), host)
    return engine


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type(OperationalError),
    reraise=True,
)
def init_db(engine: Engine) -> None:
    """Create all tables (idempotent). Retries on connection failure."""
    Base.metadata.create_all(engine)
    logger.info("Database tables ensured")


def make_session(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to the given engine."""
    return sessionmaker(bind=engine)
