"""FastAPI application entry point."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config.settings import settings
from app.database.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # ── Startup ───────────────────────────────────────────────────────
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.COMMENTARY_DIR, exist_ok=True)
    os.makedirs(settings.HF_CACHE_DIR, exist_ok=True)
    await init_db()
    print("✅  Database initialised, directories ready.")
    yield
    # ── Shutdown ──────────────────────────────────────────────────────
    print("🛑  Shutting down.")


app = FastAPI(
    title="Football Analysis Platform",
    description="AI-powered match analysis, advanced metrics (xT, xG, VAEP), and commentary generation.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving for uploads & commentary videos ───────────────────
# Created lazily so first request works even if dirs don't exist at import time


# ── Routers ───────────────────────────────────────────────────────────────
from app.routes import upload, matches, players, lineups, embeddings, analysis  # noqa: E402

app.include_router(upload.router, tags=["Upload"])
app.include_router(matches.router, tags=["Matches"])
app.include_router(players.router, tags=["Players"])
app.include_router(lineups.router, tags=["Lineups"])
app.include_router(embeddings.router, tags=["Embeddings"])
app.include_router(analysis.router, tags=["Analysis"])


@app.get("/health")
async def health():
    from app.config.gpu_config import get_device_info
    return {"status": "ok", "gpu": get_device_info()}
