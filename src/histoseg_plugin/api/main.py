import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from histoseg_plugin.api.logging import setup_logging
from histoseg_plugin.api.routes.jobs import router as jobs_router
from histoseg_plugin.api.routes.queue import router as queue_router
from histoseg_plugin.api.routes.results import router as results_router
from histoseg_plugin.api.routes.segment import router as segmentation_router
from histoseg_plugin.db.engine import (
    create_db_engine,
    create_session_factory,
)
from histoseg_plugin.db.setup import prepare_or_check_db
from histoseg_plugin.jobs.queue_service import QueueService
from histoseg_plugin.results.service import ResultService
from histoseg_plugin.settings import get_settings

setup_logging(level="DEBUG")
logger = logging.getLogger(__name__)

ALLOWED_ROOTS = [
    Path("/mnt/nas6"),
    Path("/mnt/nas7"),
    Path("/home/val/data"),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    engine = create_db_engine(settings.database_url)
    prepare_or_check_db(engine)

    session_factory = create_session_factory(engine)

    app.state.queue_service = QueueService(session_factory)
    app.state.result_service = ResultService(session_factory)
    app.state.allowed_roots = settings.allowed_roots

    if settings.debug:
        logger.info("Waiting for debugger attach...")
        import debugpy

        debugpy.listen(("0.0.0.0", 5678))
        logger.info("debugpy listening on 0.0.0.0:5678")

    try:
        yield
    finally:
        logger.info("FastAPI shutdown")
        engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(title="histoseg", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(segmentation_router)
    app.include_router(jobs_router)
    app.include_router(results_router)
    app.include_router(queue_router)
    return app


app = create_app()
