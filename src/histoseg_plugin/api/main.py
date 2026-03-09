# TODO: add an env or config handling for the app
from pathlib import Path

from fastapi.concurrency import asynccontextmanager
import torch
from fastapi import FastAPI

from histoseg_plugin.api.routes.segment import router as segmentation_router
from histoseg_plugin.models.loader import load_model_bundle

MODEL_DIR = Path("/home/valentin/workspaces/histoseg-plugin/models/models/twotasks")


@asynccontextmanager
async def lifespan(app: FastAPI):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bundle = load_model_bundle(MODEL_DIR, device=device)

    app.state.device = device
    app.state.model_bundle = bundle

    yield


def create_app() -> FastAPI:
    app = FastAPI(title="histoseg", lifespan=lifespan)
    app.include_router(segmentation_router)
    return app


app = create_app()
