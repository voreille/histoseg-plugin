from histoseg_plugin.db.engine import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from histoseg_plugin.db.migrations.runner import check_db_is_current
from histoseg_plugin.jobs.worker import run_worker_forever
from histoseg_plugin.settings import get_settings


def main() -> None:
    settings = get_settings()

    engine = create_db_engine(settings.queue_db_url)
    check_db_is_current(engine)
    init_db(engine)
    session_factory = create_session_factory(engine)

    try:
        run_worker_forever(
            settings=settings,
            session_factory=session_factory,
        )
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
