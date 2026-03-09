"""FastAPI application entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI

from fastapi_app.api.routers.video import router as video_router
from fastapi_app.services.dependencies import get_job_service

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Football Analysis API",
        version="1.0.0",
        description="Async FastAPI wrapper for the football analysis pipeline.",
    )

    app.include_router(video_router, prefix="/api/v1/video", tags=["video"])

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "football-analysis-api", "version": "1.0.0"}

    @app.on_event("startup")
    def on_startup() -> None:
        logger.info("Football Analysis API started")

    @app.on_event("shutdown")
    def on_shutdown() -> None:
        get_job_service().shutdown()
        logger.info("Football Analysis API shutdown complete")

    return app


app = create_app()

