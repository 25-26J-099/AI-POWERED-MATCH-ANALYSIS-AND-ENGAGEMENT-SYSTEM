"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Central application settings — all values configurable via env vars."""

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./football_analysis.db",
        description="Async database URL. Use postgresql+asyncpg://... for production.",
    )

    # ── File Storage ──────────────────────────────────────────────────────
    UPLOAD_DIR: str = Field(default="./uploads", description="Directory for uploaded videos")
    COMMENTARY_DIR: str = Field(default="./commentary_videos", description="Directory for commentary output videos")

    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key for expert analysis")
    OPENAI_MODEL: str = Field(default="gpt-4o", description="OpenAI model to use")

    # ── HuggingFace Model Repos ───────────────────────────────────────────
    HF_XG_REPO: str = Field(default="your-org/xg-model", description="HuggingFace repo for xG model")
    HF_VAEP_SCORING_REPO: str = Field(default="your-org/vaep-scoring-model", description="HuggingFace repo for VAEP scoring model")
    HF_VAEP_CONCEDING_REPO: str = Field(default="your-org/vaep-conceding-model", description="HuggingFace repo for VAEP conceding model")
    HF_STYLE_SCALER_REPO: str = Field(default="your-org/style-scaler", description="HuggingFace repo for style scaler")
    HF_STYLE_AUTOENCODER_REPO: str = Field(default="your-org/style-autoencoder", description="HuggingFace repo for style autoencoder")
    HF_STYLE_KMEANS_REPO: str = Field(default="your-org/style-kmeans", description="HuggingFace repo for style KMeans model")
    HF_CACHE_DIR: str = Field(default="./model_cache", description="Local cache directory for HuggingFace models")

    # ── GPU / Deployment ──────────────────────────────────────────────────
    GPU_PROVIDER: Optional[str] = Field(default=None, description="GPU provider: 'vastai', 'runpod', or None for local")
    VASTAI_API_KEY: Optional[str] = Field(default=None, description="vast.ai API key")
    RUNPOD_API_KEY: Optional[str] = Field(default=None, description="runpod API key")
    FORCE_CPU: bool = Field(default=False, description="Force CPU even if CUDA is available")

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
