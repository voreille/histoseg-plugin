from histoseg_plugin.jobs.db import init_db, get_engine
from histoseg_plugin.jobs.queue_models import Base
from histoseg_plugin.jobs.worker import run_worker_forever
from histoseg_plugin.settings import get_settings


if __name__ == "__main__":
    settings = get_settings()
    init_db(settings)
    Base.metadata.create_all(bind=get_engine())
    run_worker_forever(settings)