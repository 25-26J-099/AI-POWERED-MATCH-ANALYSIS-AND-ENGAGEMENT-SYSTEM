"""FastAPI application entrypoint for the merged full-stack system."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config.settings import settings
from app.database.database import init_db
from app.middleware import RateLimitMiddleware, RequestContextMiddleware, SecurityHeadersMiddleware
from app.routes import (
    analysis,
    analytics,
    commentary, # Added commentary router
    commentary_export,
    embeddings,
    lineups,
    matches,
    players,
    upload,
    video,
)
from app.services.dependencies import get_job_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.COMMENTARY_DIR, exist_ok=True)
    os.makedirs(settings.HF_CACHE_DIR, exist_ok=True)
    await init_db()
    logger.info("Merged football analysis API started")
    yield
    get_job_service().shutdown()
    logger.info("Merged football analysis API shutdown complete")


def create_app() -> FastAPI:
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.COMMENTARY_DIR, exist_ok=True)
    os.makedirs(settings.HF_CACHE_DIR, exist_ok=True)

    app = FastAPI(
        title="Football Analysis Platform",
        description="Merged tracking, analytics, and frontend backend for football analysis.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)

    app.include_router(upload.router, tags=["Upload"])
    app.include_router(matches.router, tags=["Matches"])
    app.include_router(players.router, tags=["Players"])
    app.include_router(analytics.router, tags=["Analytics"])
    app.include_router(commentary_export.router, tags=["Commentary Export"])
    app.include_router(lineups.router, tags=["Lineups"])
    app.include_router(embeddings.router, tags=["Embeddings"])
    app.include_router(analysis.router, tags=["AI Analysis"])
    app.include_router(video.router, prefix="/api/v1/video", tags=["Tracking"])
    app.include_router(commentary.router, tags=["Commentary"]) # Registered commentary router

    static_dirs = {
        "/static/uploads": Path(settings.UPLOAD_DIR),
        "/static/commentary": Path(settings.COMMENTARY_DIR),
    }
    for mount_path, directory in static_dirs.items():
        app.mount(mount_path, StaticFiles(directory=str(directory)), name=mount_path.replace("/", "_"))

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "service": "football-analysis-platform",
            "version": "1.0.0",
            "enterprise_features": {
                "request_ids": True,
                "security_headers": settings.SECURITY_ENABLE_HEADERS,
                "rate_limiting": {
                    "enabled": True,
                    "limit": settings.RATE_LIMIT_REQUESTS,
                    "window_seconds": settings.RATE_LIMIT_WINDOW_SECONDS,
                },
                "api_key_protection": settings.ENTERPRISE_ENFORCE_API_KEY,
            },
        }

    return app


app = create_app()
