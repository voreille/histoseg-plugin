from histoseg_plugin.jobs.db import (
    create_queue_engine,
    create_session_factory,
    init_queue_db,
)
from histoseg_plugin.jobs.worker import run_worker_forever
from histoseg_plugin.settings import get_settings


def main() -> None:
    settings = get_settings()

    engine = create_queue_engine(settings.queue_db_url)
    init_queue_db(engine)
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
