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
    HF_FOOTBALL_MODELS_REPO: str = Field(
        default="AI-POWERED-FOOTBALL-SYSTEM/football-ai-models",
        description="HuggingFace repo for shared event-detection models",
    )
    HF_EVENT_DETECTOR_WEIGHTS_FILE: str = Field(
        default="event_detector_weights.pth",
        description="Filename for the ML event detector weights in the shared HuggingFace repo",
    )
    HF_FASTREID_REPO: Optional[str] = Field(
        default=None,
        description="Optional HuggingFace repo for FastReID configs/checkpoints",
    )
    HF_FASTREID_CONFIG_FILE: str = Field(
        default="football_vit.yml",
        description="Filename for the FastReID config file in the optional repo",
    )
    HF_FASTREID_WEIGHTS_FILE: str = Field(
        default="football_vit.pth",
        description="Filename for the FastReID weights file in the optional repo",
    )
    HF_ESPCN_MODEL_FILE: str = Field(
        default="ESPCN_x2.pb",
        description="Filename for the ESPCN super-resolution model in the shared HuggingFace repo",
    )
    HF_CACHE_DIR: str = Field(default="./model_cache", description="Local cache directory for HuggingFace models")

    # ── GPU / Deployment ──────────────────────────────────────────────────
    GPU_PROVIDER: Optional[str] = Field(default=None, description="GPU provider: 'vastai', 'runpod', or None for local")
    VASTAI_API_KEY: Optional[str] = Field(default=None, description="vast.ai API key")
    RUNPOD_API_KEY: Optional[str] = Field(default=None, description="runpod API key")
    FORCE_CPU: bool = Field(default=False, description="Force CPU even if CUDA is available")

    REID_BACKEND_PRIORITY: str = Field(
        default="fastreid,torchreid,handcrafted",
        description="Comma-separated Re-ID backend priority list",
    )
    FASTREID_ENABLED: bool = Field(default=True, description="Allow FastReID backend initialization")
    FASTREID_STRICT: bool = Field(
        default=False,
        description="Fail startup when FastReID is required but unavailable",
    )
    FASTREID_DEVICE: str = Field(
        default="auto",
        description="Preferred FastReID device: auto, cpu, or cuda",
    )
    FASTREID_CONFIG_PATH: str = Field(
        default="./models/reid/fastreid/configs/football_vit.yml",
        description="Project-local default FastReID config path",
    )
    FASTREID_WEIGHTS_PATH: str = Field(
        default="./models/reid/fastreid/weights/football_vit.pth",
        description="Project-local default FastReID checkpoint path",
    )

    ENABLE_GNN_TACTICAL_ANALYSIS: bool = Field(default=True, description="Enable GNN-first tactical inference")
    GNN_MODEL_PATH: str = Field(default="./checkpoints/tactical_gnn/model.pt", description="Checkpoint path for tactical GNN inference")
    GNN_EDGE_STRATEGY: str = Field(default="knn", description="Graph edge strategy: knn or radius")
    GNN_K_NEIGHBORS: int = Field(default=4, description="k for k-nearest-neighbor graph construction")
    GNN_RADIUS: float = Field(default=18.0, description="Radius used when GNN_EDGE_STRATEGY=radius")
    GNN_DEVICE: str = Field(default="cpu", description="Device for tactical GNN inference: cpu, cuda, or auto")
    GNN_CONFIDENCE_THRESHOLD: float = Field(default=0.4, description="Confidence threshold before emitting unknown tactical labels")
    GNN_USE_HEURISTIC_FALLBACK: bool = Field(default=True, description="Fall back to heuristic tactical analysis when GNN inference is unavailable")

    COMMENTARY_LEVEL: str = Field(default="Intermediate", description="Default adaptive commentary level")
    COMMENTARY_VERBOSITY: str = Field(default="medium", description="Default commentary verbosity: low, medium, or high")
    COMMENTARY_EDUCATIONAL_MODE: bool = Field(default=False, description="Default educational mode for commentary")
    COMMENTARY_STYLE: str = Field(default="neutral", description="Default commentary tone/style")

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = Field(default=["http://localhost:5173", "http://localhost:3000"])

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
