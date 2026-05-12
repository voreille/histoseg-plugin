VERSION = 1
NAME = "task_dashboard_fields"


def _get_columns(conn, table_name: str) -> set[str]:
    rows = conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def upgrade(conn) -> None:
    columns = _get_columns(conn, "tasks")

    if "priority" not in columns:
        conn.exec_driver_sql(
            "ALTER TABLE tasks ADD COLUMN priority INTEGER NOT NULL DEFAULT 0"
        )

    if "progress_message" not in columns:
        conn.exec_driver_sql(
            "ALTER TABLE tasks ADD COLUMN progress_message VARCHAR(255)"
        )

    if "cancel_requested" not in columns:
        conn.exec_driver_sql(
            "ALTER TABLE tasks ADD COLUMN cancel_requested BOOLEAN NOT NULL DEFAULT 0"
        )

    if "created_at" not in columns:
        conn.exec_driver_sql(
            "ALTER TABLE tasks ADD COLUMN created_at DATETIME"
        )
        conn.exec_driver_sql(
            "UPDATE tasks SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL"
        )

    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_tasks_priority ON tasks (priority)"
    )

    conn.exec_driver_sql(
        """
        CREATE INDEX IF NOT EXISTS ix_tasks_priority_created_at
        ON tasks (priority, created_at)
        """
    )
