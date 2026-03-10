"""Load pre-trained models from HuggingFace repos — cached locally."""

import os
import pickle
import joblib
from typing import Optional, Tuple, Any
from app.config.settings import settings

_xg_model = None
_vaep_scoring_model = None
_vaep_conceding_model = None
_style_scaler = None
_style_autoencoder = None
_style_kmeans = None
_models_loaded = False


def _download_from_hf(repo_id: str, filename: str) -> Optional[str]:
    """Download a file from HuggingFace Hub, return local path."""
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=settings.HF_CACHE_DIR,
        )
    except Exception as e:
        print(f"⚠️  Could not download {filename} from {repo_id}: {e}")
        return None


def _load_pickle_model(repo_id: str, filename: str = "model.pkl") -> Optional[Any]:
    """Load a pickle/joblib model from HuggingFace."""
    path = _download_from_hf(repo_id, filename)
    if path is None:
        return None
    try:
        return joblib.load(path)
    except Exception:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  Failed to load model from {path}: {e}")
            return None


def _load_torch_model(repo_id: str, filename: str = "model.pth") -> Optional[Any]:
    """Load a PyTorch model from HuggingFace."""
    path = _download_from_hf(repo_id, filename)
    if path is None:
        return None
    try:
        import torch
        from app.config.gpu_config import get_device
        model = torch.load(path, map_location=get_device(), weights_only=False)
        return model
    except Exception as e:
        print(f"⚠️  Failed to load torch model from {path}: {e}")
        return None


def load_xg_model():
    """Load the xG logistic regression model."""
    global _xg_model
    if _xg_model is None:
        _xg_model = _load_pickle_model(settings.HF_XG_REPO, "xg_model.pkl")
    return _xg_model


def load_vaep_models() -> Tuple[Optional[Any], Optional[Any]]:
    """Load both VAEP models (scoring probability, conceding probability)."""
    global _vaep_scoring_model, _vaep_conceding_model
    if _vaep_scoring_model is None:
        _vaep_scoring_model = _load_pickle_model(settings.HF_VAEP_SCORING_REPO, "vaep_scoring.pkl")
    if _vaep_conceding_model is None:
        _vaep_conceding_model = _load_pickle_model(settings.HF_VAEP_CONCEDING_REPO, "vaep_conceding.pkl")
    return _vaep_scoring_model, _vaep_conceding_model


def load_style_models() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    """Load the 3 style embedding models: Scaler, StyleAutoencoder, KMeans."""
    global _style_scaler, _style_autoencoder, _style_kmeans
    if _style_scaler is None:
        _style_scaler = _load_pickle_model(settings.HF_STYLE_SCALER_REPO, "scaler.pkl")
    if _style_autoencoder is None:
        _style_autoencoder = _load_torch_model(settings.HF_STYLE_AUTOENCODER_REPO, "style_autoencoder.pth")
    if _style_kmeans is None:
        _style_kmeans = _load_pickle_model(settings.HF_STYLE_KMEANS_REPO, "kmeans.pkl")
    return _style_scaler, _style_autoencoder, _style_kmeans


def preload_all_models():
    """Pre-load all models at startup (optional)."""
    print("📦  Pre-loading models from HuggingFace...")
    load_xg_model()
    load_vaep_models()
    load_style_models()
    print("✅  All models loaded.")
