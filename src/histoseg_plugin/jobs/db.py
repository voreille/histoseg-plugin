# jobs/db.py
from __future__ import annotations

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

_engine = None
_SessionLocal = None


def init_db(settings) -> None:
    global _engine, _SessionLocal

    connect_args = {}
    if settings.queue_db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    _engine = create_engine(
        settings.queue_db_url,
        connect_args=connect_args,
        future=True,
    )
    _SessionLocal = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        future=True,
    )


def get_engine():
    if _engine is None:
        raise RuntimeError("Database not initialized. Call init_db(settings) first.")
    return _engine


@contextmanager
def get_session():
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db(settings) first.")
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()