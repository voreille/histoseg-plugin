from sqlalchemy.engine import Engine

from histoseg_plugin.db.engine import init_db
from histoseg_plugin.db.migrations.runner import (
    check_db_is_current,
    run_migrations,
    stamp_migrations,
)


def is_empty_database(engine: Engine) -> bool:
    with engine.begin() as conn:
        if engine.dialect.name == "sqlite":
            rows = conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        else:
            rows = conn.exec_driver_sql(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            ).fetchall()

    return len(rows) == 0


def prepare_or_check_db(engine: Engine) -> None:
    if is_empty_database(engine):
        init_db(engine)
        # ORM already creates the current schema; just record migrations as applied.
        stamp_migrations(engine)
        return

    check_db_is_current(engine)
