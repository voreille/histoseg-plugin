# TODO: add an env or config handling for the app
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from histoseg_plugin.api.logging import setup_logging
from histoseg_plugin.api.routes.segment import router as segmentation_router
from histoseg_plugin.models.loader import load_model_bundle

MODEL_DIR = Path("/home/valentin/workspaces/pathseg-benchmark/models/spinning-peach-46")

setup_logging(level="DEBUG")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model bundle from %s", MODEL_DIR)

    bundle = load_model_bundle(MODEL_DIR, device=device)

    logger.info("Model loaded on device %s", device)

    app.state.device = device
    app.state.model_bundle = bundle

    yield

    logger.info("FastAPI shutdown")


def create_app() -> FastAPI:
    app = FastAPI(title="histoseg", lifespan=lifespan)
    app.include_router(segmentation_router)
    return app


app = create_app()
