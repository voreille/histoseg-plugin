from sqlalchemy.engine import Engine

from histoseg_plugin.db.engine import init_db
from histoseg_plugin.db.migrations.runner import (
    check_db_is_current,
    run_migrations,
)


def is_empty_database(engine: Engine) -> bool:
    with engine.begin() as conn:
        rows = conn.exec_driver_sql(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
              AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()

    return len(rows) == 0


def prepare_or_check_db(engine: Engine) -> None:
    if is_empty_database(engine):
        init_db(engine)
        run_migrations(engine)
        return

    check_db_is_current(engine)
