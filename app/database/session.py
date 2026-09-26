"""SQLAlchemy engine and session factory for MySQL 8."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.exceptions import DatabaseError
from app.core.logging_config import get_logger

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine() -> Engine:
    """Create or return the shared SQLAlchemy engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.sqlalchemy_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Create or return the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def check_database_connection() -> bool:
    """Return True when MySQL is reachable."""
    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("MySQL connection check failed: %s", exc)
        return False


def require_database() -> None:
    """Raise DatabaseError if MySQL is not reachable."""
    if not check_database_connection():
        raise DatabaseError(
            "Cannot connect to MySQL. Check DATABASE_URL and ensure the server is running."
        )
