import logging

import torch
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from histoseg_plugin.api.logging import setup_logging
from histoseg_plugin.api.routes.jobs import router as jobs_router
from histoseg_plugin.api.routes.queue import router as queue_router
from histoseg_plugin.api.routes.segment import router as segmentation_router
from histoseg_plugin.core.inference.loader import load_inference_bundle
from histoseg_plugin.settings import get_settings


settings = get_settings()
MODEL_DIR = settings.models_root / "giddy-spaceship-137"

setup_logging(level="DEBUG")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model bundle from %s", MODEL_DIR)

    inference_bundle = load_inference_bundle(MODEL_DIR, device=device)

    app.state.inference_bundle = inference_bundle
    app.state.device = device

    logger.info("Model loaded on device %s", device)
    if settings.debug:
        logger.info("Waiting for debugger attach...")
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("debugpy listening on 0.0.0.0:5678")

    yield

    logger.info("FastAPI shutdown")


def create_app() -> FastAPI:
    app = FastAPI(title="histoseg", lifespan=lifespan)
    app.include_router(segmentation_router)
    app.include_router(jobs_router)
    app.include_router(queue_router)
    return app


app = create_app()
