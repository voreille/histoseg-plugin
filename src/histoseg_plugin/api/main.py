from fastapi import FastAPI
from histoseg_plugin.api.routes.tissue import router as tissue_router


def create_app() -> FastAPI:
    app = FastAPI(title="histoseg")
    app.include_router(tissue_router)
    return app


app = create_app()
