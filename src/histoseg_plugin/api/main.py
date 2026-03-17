# TODO: add an env or config handling for the app
import logging
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager

from histoseg_plugin.api.logging import setup_logging
from histoseg_plugin.api.routes.segment import router as segmentation_router
from histoseg_plugin.core.model_runtime.loader import load_model_bundle

MODEL_DIR = Path("/home/valentin/workspaces/histoseg-plugin/models/AIgrading_anorak")
# MODEL_DIR = Path("/home/valentin/workspaces/ignite-data-toolkit/data/models/he_export")
# MODEL_DIR = Path("/home/valentin/workspaces/histoseg-plugin/models/models/spinning-peach-46")

setup_logging(level="DEBUG")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model bundle from %s", MODEL_DIR)

    runner = load_model_bundle(MODEL_DIR, device=device)

    app.state.model_runner = runner
    app.state.device = device

    logger.info("Model loaded on device %s", device)

    yield

    logger.info("FastAPI shutdown")


def create_app() -> FastAPI:
    app = FastAPI(title="histoseg", lifespan=lifespan)
    app.include_router(segmentation_router)
    return app


app = create_app()
