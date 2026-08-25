"""
Migrate data from the SQLite queue DB to PostgreSQL.

Usage (from inside the dev container, or locally with both DBs reachable):

    python scripts/migrate_sqlite_to_pg.py \
        data/histoseg_queue.db \
        "postgresql+psycopg2://histoseg:histoseg@localhost:5432/histoseg"

Run AFTER the app has started at least once against the target PostgreSQL
instance so that tables and enum types already exist.
"""

import argparse
import sqlite3
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import execute_values


def _parse_dt(val: str | None) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    # Try ISO 8601 first (handles offset-aware strings like '2026-06-16T11:55:04+00:00')
    try:
        dt = datetime.fromisoformat(val)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    # Fallback for SQLite's space-separated format without timezone
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unrecognised datetime format: {val!r}")


# SQLite stores datetimes as plain strings; these columns need conversion.
_DT_COLS: dict[str, set[str]] = {
    "jobs": {"created_at", "updated_at"},
    "tasks": {"created_at", "heartbeat_at", "started_at", "finished_at"},
    "results": {"created_at"},
    "schema_migrations": {"applied_at"},
}

# SQLite stores booleans as integers (0/1); PostgreSQL requires actual bool.
_BOOL_COLS: dict[str, set[str]] = {
    "tasks": {"cancel_requested"},
    "queue_state": {"paused"},
}

# Copy order must respect FK constraints (parents before children).
_TABLES = ["queue_state", "jobs", "results", "tasks", "schema_migrations"]


def _col_names(src: sqlite3.Connection, table: str) -> list[str]:
    cur = src.execute(f"SELECT * FROM {table} LIMIT 0")
    return [d[0] for d in cur.description]


def migrate(sqlite_path: str, pg_url: str, *, dry_run: bool = False) -> None:
    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row

    # Strip SQLAlchemy prefix so psycopg2 can parse the DSN.
    dsn = pg_url.replace("postgresql+psycopg2://", "postgresql://")
    dst = psycopg2.connect(dsn)
    cur = dst.cursor()

    # Clear target tables in child-first order to satisfy FK constraints.
    print("Clearing target tables …")
    cur.execute(
        "TRUNCATE TABLE tasks, results, jobs, queue_state, schema_migrations "
        "RESTART IDENTITY CASCADE"
    )

    for table in _TABLES:
        rows = src.execute(f"SELECT * FROM {table}").fetchall()
        if not rows:
            print(f"  {table}: empty, skipping")
            continue

        cols = _col_names(src, table)
        dt_cols = _DT_COLS.get(table, set())
        bool_cols = _BOOL_COLS.get(table, set())

        records = []
        for row in rows:
            record = []
            for col, val in zip(cols, row):
                if col in dt_cols:
                    val = _parse_dt(val)
                elif col in bool_cols and val is not None:
                    val = bool(val)
                # A task left in 'running' state is stale — treat as interrupted.
                if table == "tasks" and col == "status" and val == "running":
                    val = "interrupted"
                record.append(val)
            records.append(tuple(record))

        if dry_run:
            print(f"  {table}: {len(records)} rows (dry-run, not inserted)")
            continue

        col_sql = ", ".join(cols)
        execute_values(cur, f"INSERT INTO {table} ({col_sql}) VALUES %s", records)

        # Reset the serial sequence so future inserts don't collide.
        if "id" in cols:
            cur.execute(
                f"SELECT setval(pg_get_serial_sequence('{table}', 'id'), MAX(id)) FROM {table}"
            )

        print(f"  {table}: {len(records)} rows migrated")

    if not dry_run:
        dst.commit()
        print("Committed.")
    else:
        print("Dry-run complete — no changes written.")

    src.close()
    dst.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_path", help="Path to the SQLite .db file")
    parser.add_argument("pg_url", help="PostgreSQL connection URL")
    parser.add_argument(
        "--dry-run", action="store_true", help="Read only, do not write to PostgreSQL"
    )
    args = parser.parse_args()
    migrate(args.sqlite_path, args.pg_url, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
