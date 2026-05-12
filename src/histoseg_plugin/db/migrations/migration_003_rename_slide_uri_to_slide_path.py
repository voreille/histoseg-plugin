VERSION = 3
NAME = "rename_slide_uri_to_slide_path"


def _table_exists(conn, table_name: str) -> bool:
    row = conn.exec_driver_sql(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table' AND name=?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def _get_columns(conn, table_name: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def _rename_or_backfill_column(
    conn,
    *,
    table_name: str,
    old_name: str,
    new_name: str,
    column_type: str = "TEXT",
) -> None:
    if not _table_exists(conn, table_name):
        return

    columns = _get_columns(conn, table_name)

    old_exists = old_name in columns
    new_exists = new_name in columns

    if old_exists and not new_exists:
        # SQLite >= 3.25 supports this.
        conn.exec_driver_sql(
            f"ALTER TABLE {table_name} RENAME COLUMN {old_name} TO {new_name}"
        )
        return

    if old_exists and new_exists:
        # Defensive case: both columns exist.
        # Keep old column but ensure new column is populated.
        conn.exec_driver_sql(
            f"""
            UPDATE {table_name}
            SET {new_name} = {old_name}
            WHERE {new_name} IS NULL
            """
        )
        return

    if not old_exists and not new_exists:
        # Fresh/partial schema edge case.
        conn.exec_driver_sql(
            f"ALTER TABLE {table_name} ADD COLUMN {new_name} {column_type}"
        )


def upgrade(conn) -> None:
    _rename_or_backfill_column(
        conn,
        table_name="tasks",
        old_name="slide_uri",
        new_name="slide_path",
        column_type="TEXT",
    )

    _rename_or_backfill_column(
        conn,
        table_name="results",
        old_name="slide_uri",
        new_name="slide_path",
        column_type="TEXT",
    )
