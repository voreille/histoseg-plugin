# src/histoseg_plugin/db/migrations/runner.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

from sqlalchemy.engine import Engine

from histoseg_plugin.db.migrations import (
    migration_001_task_dashboard_fields,
    migration_002_results_and_enum_values,
    migration_003_rename_slide_uri_to_slide_path,
)


class Migration(Protocol):
    VERSION: int
    NAME: str

    def upgrade(self, conn) -> None: ...


MIGRATIONS = [
    migration_001_task_dashboard_fields,
    migration_002_results_and_enum_values,
    migration_003_rename_slide_uri_to_slide_path,
]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_schema_migrations_table(conn) -> None:
    conn.exec_driver_sql(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            applied_at DATETIME NOT NULL
        )
        """
    )


def get_applied_versions(conn) -> set[int]:
    ensure_schema_migrations_table(conn)

    rows = conn.exec_driver_sql("SELECT version FROM schema_migrations").fetchall()

    return {int(row[0]) for row in rows}


def get_current_version(engine: Engine) -> int:
    with engine.begin() as conn:
        ensure_schema_migrations_table(conn)

        row = conn.exec_driver_sql(
            "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
        ).one()

        return int(row[0])


def get_latest_version() -> int:
    if not MIGRATIONS:
        return 0

    return max(migration.VERSION for migration in MIGRATIONS)


def run_migrations(engine: Engine) -> None:
    with engine.begin() as conn:
        ensure_schema_migrations_table(conn)
        applied_versions = get_applied_versions(conn)

        for migration in sorted(MIGRATIONS, key=lambda m: m.VERSION):
            if migration.VERSION in applied_versions:
                continue

            migration.upgrade(conn)

            conn.exec_driver_sql(
                """
                INSERT INTO schema_migrations (version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (
                    migration.VERSION,
                    migration.NAME,
                    _utcnow_iso(),
                ),
            )


def check_db_is_current(engine: Engine) -> None:
    current_version = get_current_version(engine)
    latest_version = get_latest_version()

    if current_version < latest_version:
        raise RuntimeError(
            f"Database schema is version {current_version}, "
            f"but this plugin requires version {latest_version}. "
            "Run the database migration command before starting the API/worker."
        )

    if current_version > latest_version:
        raise RuntimeError(
            f"Database schema is version {current_version}, "
            f"but this plugin only knows migrations up to {latest_version}. "
            "You may be running an older plugin version against a newer database."
        )
