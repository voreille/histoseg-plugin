# src/histoseg_plugin/db/migrate.py

from __future__ import annotations

import argparse

from histoseg_plugin.db.engine import create_db_engine, init_db
from histoseg_plugin.db.migrations.runner import (
    check_db_is_current,
    get_current_version,
    get_latest_version,
    run_migrations,
)
from histoseg_plugin.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage histoseg queue database migrations."
    )

    parser.add_argument(
        "command",
        choices=["version", "check", "upgrade"],
        help="Migration command to run.",
    )

    parser.add_argument(
        "--db-url",
        default=None,
        help="Override queue database URL. Defaults to settings.database_url.",
    )

    args = parser.parse_args()

    settings = get_settings()
    db_url = args.db_url or settings.database_url

    engine = create_db_engine(db_url)

    if args.command == "version":
        current = get_current_version(engine)
        latest = get_latest_version()
        print(f"Current DB version: {current}")
        print(f"Latest known version: {latest}")
        return

    if args.command == "check":
        check_db_is_current(engine)
        print("Database schema is up to date.")
        return

    if args.command == "upgrade":
        before = get_current_version(engine)
        latest = get_latest_version()

        print(f"Current DB version: {before}")
        print(f"Latest known version: {latest}")

        # Important for fresh/local databases
        init_db(engine)

        run_migrations(engine)

        after = get_current_version(engine)
        print(f"Database upgraded to version: {after}")
        return


if __name__ == "__main__":
    main()
