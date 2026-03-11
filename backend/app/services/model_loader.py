"""Load and cache analytics models from HuggingFace."""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib

from app.config.settings import settings

logger = logging.getLogger(__name__)

_UNSET = object()
_FAILED = object()

_xg_model: Any = _UNSET
_vaep_scoring_model: Any = _UNSET
_vaep_conceding_model: Any = _UNSET
_style_scaler: Any = _UNSET
_style_autoencoder: Any = _UNSET
_style_kmeans: Any = _UNSET


class StyleAutoEncoderModule:  # pragma: no cover - thin wrapper around torch
    """Runtime copy of the training notebook autoencoder architecture."""

    def __init__(self, input_dim: int, embedding_dim: int = 8):
        import torch.nn as nn

        self._module = nn.Module()
        self._module.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim),
        )
        self._module.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def __getattr__(self, item):
        return getattr(self._module, item)


def _download_from_hf(repo_id: str, filename: str) -> Optional[str]:
    if not repo_id or repo_id.startswith("your-org/") or "/" not in repo_id:
        return None
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=settings.HF_CACHE_DIR,
            force_download=False,
            local_files_only=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not download %s from %s: %s", filename, repo_id, exc)
        return None


def download_hf_asset(repo_id: str, filename: str) -> Optional[str]:
    """Download a raw artifact from HuggingFace and return the cached local path."""
    return _download_from_hf(repo_id, filename)


def _load_joblib_or_pickle(path: str | Path) -> Optional[Any]:
    for loader_name, loader in (
        ("joblib", lambda p: joblib.load(p)),
    ):
        try:
            return loader(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed loading %s via %s: %s", path, loader_name, exc)

    try:
        with open(path, "rb") as infile:
            return pickle.load(infile)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed loading %s via pickle: %s", path, exc)

    try:
        import cloudpickle

        with open(path, "rb") as infile:
            return cloudpickle.load(infile)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed loading %s via cloudpickle: %s", path, exc)
        return None


def _load_pickle_model(repo_id: str, filename: str) -> Optional[Any]:
    path = _download_from_hf(repo_id, filename)
    if path is None:
        return None
    return _load_joblib_or_pickle(path)


def _load_style_autoencoder(repo_id: str, filename: str = "style_autoencoder.pth") -> Optional[Any]:
    path = _download_from_hf(repo_id, filename)
    if path is None:
        return None

    try:
        import torch

        from app.config.gpu_config import get_device

        checkpoint = torch.load(path, map_location=get_device(), weights_only=False)
        input_dim = int(get_style_scaler_feature_count() or 31)
        module = StyleAutoEncoderModule(input_dim=input_dim)._module

        state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            logger.warning("Unexpected autoencoder checkpoint format at %s", path)
            return None

        module.load_state_dict(state_dict, strict=False)
        module.eval()
        return module
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load style autoencoder from %s: %s", path, exc)
        return None


def load_xg_model():
    global _xg_model
    if _xg_model is _UNSET:
        _xg_model = _load_pickle_model(settings.HF_XG_REPO, "xg_model.pkl") or _FAILED
    return None if _xg_model is _FAILED else _xg_model


def load_vaep_score_model():
    global _vaep_scoring_model
    if _vaep_scoring_model is _UNSET:
        _vaep_scoring_model = _load_pickle_model(settings.HF_VAEP_SCORING_REPO, "vaep_score_model_2.pkl") or _FAILED
        if _vaep_scoring_model is _FAILED:
            logger.warning(
                "VAEP scoring model unavailable. Install compatible runtime dependencies such as lightgbm/cloudpickle if required."
            )
    return None if _vaep_scoring_model is _FAILED else _vaep_scoring_model


def load_vaep_concede_model():
    global _vaep_conceding_model
    if _vaep_conceding_model is _UNSET:
        _vaep_conceding_model = _load_pickle_model(settings.HF_VAEP_CONCEDING_REPO, "vaep_concede_model_2.pkl") or _FAILED
        if _vaep_conceding_model is _FAILED:
            logger.warning(
                "VAEP conceding model unavailable. Install compatible runtime dependencies such as lightgbm/cloudpickle if required."
            )
    return None if _vaep_conceding_model is _FAILED else _vaep_conceding_model


def load_vaep_models() -> Tuple[Optional[Any], Optional[Any]]:
    return load_vaep_score_model(), load_vaep_concede_model()


def load_style_scaler():
    global _style_scaler
    if _style_scaler is _UNSET:
        _style_scaler = _load_pickle_model(settings.HF_STYLE_SCALER_REPO, "style_scaler.pkl") or _FAILED
    return None if _style_scaler is _FAILED else _style_scaler


def load_style_cluster_model():
    global _style_kmeans
    if _style_kmeans is _UNSET:
        _style_kmeans = _load_pickle_model(settings.HF_STYLE_KMEANS_REPO, "style_cluster_model.pkl") or _FAILED
    return None if _style_kmeans is _FAILED else _style_kmeans


def load_style_autoencoder():
    global _style_autoencoder
    if _style_autoencoder is _UNSET:
        _style_autoencoder = _load_style_autoencoder(settings.HF_STYLE_AUTOENCODER_REPO, "style_autoencoder.pth") or _FAILED
    return None if _style_autoencoder is _FAILED else _style_autoencoder


def load_style_models() -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
    return load_style_scaler(), load_style_autoencoder(), load_style_cluster_model()


def get_style_scaler_feature_names() -> list[str]:
    scaler = load_style_scaler()
    if scaler is None:
        return []
    names = getattr(scaler, "feature_names_in_", None)
    if names is not None:
        return [str(name) for name in names]
    n_features = getattr(scaler, "n_features_in_", 0)
    return [f"feature_{index}" for index in range(int(n_features))]


def get_style_scaler_feature_count() -> int:
    scaler = load_style_scaler()
    if scaler is None:
        return 0
    n_features = getattr(scaler, "n_features_in_", None)
    if n_features is not None:
        return int(n_features)
    names = getattr(scaler, "feature_names_in_", None)
    return len(names) if names is not None else 0


def preload_all_models():
    logger.info("Pre-loading analytics models...")
    load_xg_model()
    load_vaep_models()
    load_style_models()
    logger.info("Analytics models loaded")
