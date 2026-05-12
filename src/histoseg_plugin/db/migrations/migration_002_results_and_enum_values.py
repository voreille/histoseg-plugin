VERSION = 2
NAME = "results_table_and_lowercase_enum_values"


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


def upgrade(conn) -> None:
    # 1. Create results table if missing
    if not _table_exists(conn, "results"):
        conn.exec_driver_sql(
            """
            CREATE TABLE results (
                id INTEGER PRIMARY KEY,
                task_hash VARCHAR(64) NOT NULL,
                slide_path TEXT NOT NULL,
                model_id VARCHAR(128) NOT NULL,
                result_dir TEXT NOT NULL,
                geojson_path TEXT,
                stats_path TEXT,
                created_at DATETIME NOT NULL
            )
            """
        )

    conn.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_results_task_hash ON results (task_hash)"
    )

    # 2. Ensure tasks.result_id exists if older DB did not have it
    task_columns = _get_columns(conn, "tasks")
    if "result_id" not in task_columns:
        conn.exec_driver_sql(
            "ALTER TABLE tasks ADD COLUMN result_id INTEGER"
        )
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_tasks_result_id ON tasks (result_id)"
        )

    # 3. Migrate enum names to enum values, if needed
    # Jobs
    if _table_exists(conn, "jobs"):
        conn.exec_driver_sql("UPDATE jobs SET status = 'pending' WHERE status = 'PENDING'")
        conn.exec_driver_sql("UPDATE jobs SET status = 'running' WHERE status = 'RUNNING'")
        conn.exec_driver_sql("UPDATE jobs SET status = 'paused' WHERE status = 'PAUSED'")
        conn.exec_driver_sql("UPDATE jobs SET status = 'completed' WHERE status = 'COMPLETED'")
        conn.exec_driver_sql("UPDATE jobs SET status = 'failed' WHERE status = 'FAILED'")
        conn.exec_driver_sql("UPDATE jobs SET status = 'partial' WHERE status = 'PARTIAL'")

    # Tasks
    if _table_exists(conn, "tasks"):
        conn.exec_driver_sql("UPDATE tasks SET status = 'pending' WHERE status = 'PENDING'")
        conn.exec_driver_sql("UPDATE tasks SET status = 'running' WHERE status = 'RUNNING'")
        conn.exec_driver_sql("UPDATE tasks SET status = 'completed' WHERE status = 'COMPLETED'")
        conn.exec_driver_sql("UPDATE tasks SET status = 'failed' WHERE status = 'FAILED'")
        conn.exec_driver_sql("UPDATE tasks SET status = 'interrupted' WHERE status = 'INTERRUPTED'")
        conn.exec_driver_sql("UPDATE tasks SET status = 'cached' WHERE status = 'CACHED'")
        conn.exec_driver_sql("UPDATE tasks SET status = 'cancelled' WHERE status = 'CANCELLED'")
