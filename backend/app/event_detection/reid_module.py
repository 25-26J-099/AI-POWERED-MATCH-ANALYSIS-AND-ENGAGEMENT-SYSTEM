"""
Shared Re-ID backend resolution and embedding utilities.

Backend priority:
1. FastReID with ViT checkpoint/config when available
2. torchreid with a strong pretrained model such as OSNet
3. Hand-crafted appearance features as a final compatibility fallback
"""
from __future__ import annotations

import importlib.util
import contextlib
import io
import logging
import os
import warnings
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from app.services.model_loader import download_hf_asset
from app.utils.runtime_compat import apply_runtime_compatibility_shims

try:
    import cv2
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    Image = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal runtimes
    torch = None

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _normalize_vector(vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    arr = np.asarray(vector, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return arr / norm


class ReIDModel:
    """Shared embedding extractor for football-player re-identification."""

    def __init__(self, model_path: Optional[str] = None, config=None):
        self.config = config
        self.model_path = model_path or getattr(config, "model_path", "") or ""
        self.fastreid_config_path = getattr(config, "fastreid_config_path", "") or ""
        self.torchreid_model_name = getattr(config, "torchreid_model_name", "osnet_ain_x1_0")
        self.crop_width = int(getattr(config, "crop_width", 128))
        self.crop_height = int(getattr(config, "crop_height", 256))
        self.backend_priority: Sequence[str] = tuple(
            getattr(config, "backend_priority", ("fastreid", "torchreid", "handcrafted"))
        )
        self.device = self._resolve_device(getattr(config, "device", "auto"))
        self.torchreid_device = self._resolve_device(getattr(config, "torchreid_device", "cpu"))
        self.torchreid_allow_cpu = bool(getattr(config, "torchreid_allow_cpu", False))
        self.fastreid_enabled = bool(getattr(config, "fastreid_enabled", True))
        self.strict_fastreid = bool(getattr(config, "strict_fastreid", False))
        self.hf_fastreid_repo = getattr(config, "hf_fastreid_repo", "") or ""
        self.hf_fastreid_config_file = getattr(config, "hf_fastreid_config_file", "football_vit.yml")
        self.hf_fastreid_weights_file = getattr(config, "hf_fastreid_weights_file", "football_vit.pth")

        self.backend = "uninitialized"
        self.model = None
        self.feature_extractor = None
        self.backend_reason = "uninitialized"
        self.backend_history: list[dict[str, object]] = []
        self.resolved_config_path = self._resolve_path(self.fastreid_config_path)
        self.resolved_weights_path = self._resolve_path(self.model_path)

        self._initialize_backend()

    def _resolve_device(self, preferred: str) -> str:
        if preferred not in {"auto", "cpu", "cuda"}:
            preferred = "auto"
        if preferred == "auto":
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        if preferred == "cuda" and (torch is None or not torch.cuda.is_available()):
            return "cpu"
        return preferred

    def _resolve_path(self, value: str) -> Optional[Path]:
        if not value:
            return None
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (BACKEND_ROOT / candidate).resolve()

    def _register_backend_attempt(self, backend_name: str, available: bool, reason: str) -> None:
        self.backend_history.append(
            {
                "backend": backend_name,
                "available": available,
                "reason": reason,
            }
        )
        if not available and self.backend_reason in {"uninitialized", "handcrafted fallback engaged"}:
            self.backend_reason = reason

    def _ensure_local_asset(self, path: Optional[Path], filename: str) -> Optional[Path]:
        if path is None:
            return None
        if path.exists():
            return path
        if not self.hf_fastreid_repo:
            return path

        downloaded = download_hf_asset(self.hf_fastreid_repo, filename)
        if not downloaded:
            return path

        downloaded_path = Path(downloaded).resolve()
        self.backend_reason = f"downloaded missing FastReID asset from {self.hf_fastreid_repo}"
        return downloaded_path

    def _initialize_backend(self) -> None:
        for backend_name in self.backend_priority:
            initializer = getattr(self, f"_init_{backend_name}_backend", None)
            if initializer and initializer():
                if (
                    self.strict_fastreid
                    and "fastreid" in self.backend_priority
                    and self.backend != "fastreid"
                ):
                    raise RuntimeError(
                        f"FastReID strict mode is enabled but backend initialization failed: {self.backend_reason}"
                    )
                logger.info("[Re-ID] Using %s backend", self.backend)
                return

        self._init_handcrafted_backend()
        if self.strict_fastreid and "fastreid" in self.backend_priority:
            raise RuntimeError(
                f"FastReID strict mode is enabled but backend initialization failed: {self.backend_reason}"
            )
        logger.info("[Re-ID] Falling back to handcrafted backend: %s", self.backend_reason)

    def _init_fastreid_backend(self) -> bool:
        if not self.fastreid_enabled:
            self._register_backend_attempt("fastreid", False, "FastReID disabled by configuration")
            return False
        if torch is None:
            self._register_backend_attempt("fastreid", False, "PyTorch is unavailable")
            return False
        if importlib.util.find_spec("fastreid") is None:
            self._register_backend_attempt("fastreid", False, "fastreid package not installed")
            return False

        self.resolved_config_path = self._ensure_local_asset(
            self.resolved_config_path, self.hf_fastreid_config_file
        )
        self.resolved_weights_path = self._ensure_local_asset(
            self.resolved_weights_path, self.hf_fastreid_weights_file
        )
        if self.resolved_config_path is None:
            self._register_backend_attempt("fastreid", False, "FastReID config path is not configured")
            return False
        if self.resolved_weights_path is None:
            self._register_backend_attempt("fastreid", False, "FastReID weights path is not configured")
            return False
        if not self.resolved_config_path.exists():
            self._register_backend_attempt(
                "fastreid",
                False,
                f"FastReID config missing at {self.resolved_config_path}",
            )
            return False
        if not self.resolved_weights_path.exists():
            self._register_backend_attempt(
                "fastreid",
                False,
                f"FastReID weights missing at {self.resolved_weights_path}",
            )
            return False

        try:
            apply_runtime_compatibility_shims()
            from fastreid.config import get_cfg
            from fastreid.engine import DefaultPredictor

            cfg = get_cfg()
            cfg.merge_from_file(str(self.resolved_config_path))
            cfg.MODEL.WEIGHTS = str(self.resolved_weights_path)
            cfg.MODEL.DEVICE = self.device
            self.model = DefaultPredictor(cfg)
            self.backend = "fastreid"
            self.backend_reason = "FastReID initialized successfully"
            self._register_backend_attempt("fastreid", True, self.backend_reason)
            return True
        except Exception as exc:  # noqa: BLE001 - backend fallback must be resilient
            self.model = None
            self._register_backend_attempt("fastreid", False, f"FastReID init failed: {exc}")
            logger.warning("[Re-ID] FastReID backend unavailable: %s", exc)
            return False

    def _init_torchreid_backend(self) -> bool:
        if self.torchreid_device == "cpu" and not self.torchreid_allow_cpu:
            self._register_backend_attempt(
                "torchreid",
                False,
                "torchreid CPU execution disabled for live pipeline performance",
            )
            return False
        if importlib.util.find_spec("torchreid") is None:
            self._register_backend_attempt("torchreid", False, "torchreid package not installed")
            return False
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Cython evaluation.*", category=UserWarning)
                try:
                    from torchreid.utils import FeatureExtractor
                except ImportError:
                    from torchreid.reid.utils import FeatureExtractor

            torch_cache_dir = (BACKEND_ROOT / "model_cache" / "torch").resolve()
            torch_cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("TORCH_HOME", str(torch_cache_dir))

            kwargs = {
                "model_name": self.torchreid_model_name,
                "device": self.torchreid_device,
            }
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Cython evaluation.*", category=UserWarning)
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.feature_extractor = FeatureExtractor(**kwargs)
            self.backend = "torchreid"
            self.backend_reason = f"torchreid initialized successfully on {self.torchreid_device}"
            self._register_backend_attempt("torchreid", True, self.backend_reason)
            return True
        except Exception as exc:  # noqa: BLE001 - backend fallback must be resilient
            self.feature_extractor = None
            self._register_backend_attempt("torchreid", False, f"torchreid init failed: {exc}")
            logger.warning("[Re-ID] torchreid backend unavailable: %s", exc)
            return False

    def _init_handcrafted_backend(self) -> bool:
        self.backend = "handcrafted"
        if self.backend_reason == "uninitialized":
            self.backend_reason = "handcrafted fallback engaged"
        self._register_backend_attempt("handcrafted", True, self.backend_reason)
        return True

    def crop_player(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        height, width = frame.shape[:2]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))

        if x2 - x1 < 8 or y2 - y1 < 16:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop.copy()

    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if cv2 is not None:
            return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        if Image is not None:
            pil_image = Image.fromarray(image[:, :, ::-1] if image.ndim == 3 and image.shape[2] == 3 else image)
            resized = pil_image.resize(size, Image.BILINEAR)
            resized_np = np.asarray(resized)
            if resized_np.ndim == 2:
                return np.stack([resized_np, resized_np, resized_np], axis=-1)
            if resized_np.ndim == 3 and resized_np.shape[2] == 3:
                return resized_np[:, :, ::-1]
            return resized_np
        target_w, target_h = size
        y_idx = np.linspace(0, image.shape[0] - 1, target_h).astype(int)
        x_idx = np.linspace(0, image.shape[1] - 1, target_w).astype(int)
        return image[np.ix_(y_idx, x_idx)]

    def _letterbox_resize(self, image: np.ndarray) -> np.ndarray:
        src_h, src_w = image.shape[:2]
        if src_h <= 0 or src_w <= 0:
            raise ValueError("invalid image size")

        scale = min(self.crop_width / src_w, self.crop_height / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = self._resize_image(image, (new_w, new_h))

        pad_w = self.crop_width - new_w
        pad_h = self.crop_height - new_h
        top = pad_h // 2
        left = pad_w // 2
        padded = np.zeros((self.crop_height, self.crop_width, resized.shape[2]), dtype=resized.dtype)
        padded[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        return padded

    def prepare_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        if image is None or image.size == 0:
            return None
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        return self._letterbox_resize(image)

    def _prepare_fastreid_tensor(self, image: np.ndarray):
        if torch is None:
            return None
        rgb_image = image[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb_image - mean) / std
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.from_numpy(chw).unsqueeze(0)
        if self.device == "cuda":
            tensor = tensor.cuda()
        return tensor

    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        embeddings = self.extract_embeddings([image])
        return embeddings[0] if embeddings else None

    def extract_embeddings(self, images: Sequence[np.ndarray]) -> list[Optional[np.ndarray]]:
        """Extract embeddings for a batch of player crops.

        Batched inference matters on GPUs: sending 15-25 tiny player crops one by
        one keeps the accelerator mostly idle and pays the host/device transfer
        cost repeatedly.
        """
        outputs: list[Optional[np.ndarray]] = [None] * len(images)
        prepared_images: list[np.ndarray] = []
        prepared_indices: list[int] = []

        for idx, image in enumerate(images):
            prepared = self.prepare_image(image)
            if prepared is None:
                continue
            prepared_indices.append(idx)
            prepared_images.append(prepared)

        if not prepared_images:
            return outputs

        if self.backend == "fastreid":
            embeddings = self._extract_fastreid_batch(prepared_images)
        elif self.backend == "torchreid":
            embeddings = self._extract_torchreid_batch(prepared_images)
        else:
            embeddings = [self._extract_handcrafted(image) for image in prepared_images]

        for idx, embedding in zip(prepared_indices, embeddings):
            outputs[idx] = _normalize_vector(embedding)
        return outputs

    def _extract_fastreid_batch(self, images: Sequence[np.ndarray]) -> list[Optional[np.ndarray]]:
        if self.model is None or torch is None:
            return [None] * len(images)
        try:
            tensors = [self._prepare_fastreid_tensor(image) for image in images]
            tensors = [tensor for tensor in tensors if tensor is not None]
            if len(tensors) != len(images):
                return [self._extract_fastreid(image) for image in images]

            batched = torch.cat(tensors, dim=0)
            with torch.inference_mode():
                prediction = self.model(batched)
            if isinstance(prediction, dict):
                for key in ("features", "embeddings", "outputs"):
                    if key in prediction:
                        prediction = prediction[key]
                        break
            if hasattr(prediction, "detach"):
                prediction = prediction.detach().cpu().numpy()
            arr = np.asarray(prediction, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] != len(images):
                return [self._extract_fastreid(image) for image in images]
            return [arr[idx].reshape(-1) for idx in range(arr.shape[0])]
        except Exception as exc:  # noqa: BLE001 - batched backend fallback must be resilient
            logger.warning("[Re-ID] Batched FastReID extraction failed, using single-crop fallback: %s", exc)
            return [self._extract_fastreid(image) for image in images]

    def _extract_torchreid_batch(self, images: Sequence[np.ndarray]) -> list[Optional[np.ndarray]]:
        if self.feature_extractor is None:
            return [None] * len(images)
        try:
            rgb_images = [image[:, :, ::-1] for image in images]
            prediction = self.feature_extractor(rgb_images)
            if hasattr(prediction, "detach"):
                prediction = prediction.detach().cpu().numpy()
            arr = np.asarray(prediction, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[0] != len(images):
                return [self._extract_torchreid(image) for image in images]
            return [arr[idx].reshape(-1) for idx in range(arr.shape[0])]
        except Exception as exc:  # noqa: BLE001 - backend fallback must be resilient
            if "cuda" in str(exc).lower():
                self.feature_extractor = None
                self.backend = "handcrafted"
                self.backend_reason = "torchreid CUDA batch extraction failed; switched to handcrafted fallback"
                logger.warning(
                    "[Re-ID] torchreid CUDA batch extraction failed; disabling torchreid and using handcrafted fallback: %s",
                    exc,
                )
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                return [self._extract_handcrafted(image) for image in images]
            logger.warning("[Re-ID] Batched torchreid extraction failed, using single-crop fallback: %s", exc)
            return [self._extract_torchreid(image) for image in images]

    def _extract_fastreid(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None or torch is None:
            return None
        try:
            batched = self._prepare_fastreid_tensor(image)
            try:
                prediction = self.model(batched)
            except Exception:
                prediction = self.model(image[:, :, ::-1])
            if isinstance(prediction, dict):
                for key in ("features", "embeddings", "outputs"):
                    if key in prediction:
                        prediction = prediction[key]
                        break
            if hasattr(prediction, "detach"):
                prediction = prediction.detach().cpu().numpy()
            embedding = np.asarray(prediction, dtype=np.float32).reshape(-1)
            return embedding
        except Exception as exc:  # noqa: BLE001 - inference fallback must be resilient
            logger.warning("[Re-ID] FastReID extraction failed, falling back to handcrafted: %s", exc)
            return self._extract_handcrafted(image)

    def _extract_torchreid(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.feature_extractor is None:
            return None
        try:
            rgb_image = image[:, :, ::-1]
            prediction = self.feature_extractor([rgb_image])
            if hasattr(prediction, "detach"):
                prediction = prediction.detach().cpu().numpy()
            embedding = np.asarray(prediction, dtype=np.float32)
            if embedding.ndim > 1:
                embedding = embedding[0]
            return embedding.reshape(-1)
        except Exception as exc:  # noqa: BLE001 - inference fallback must be resilient
            if "cuda" in str(exc).lower():
                self.feature_extractor = None
                self.backend = "handcrafted"
                self.backend_reason = "torchreid CUDA extraction failed; switched to handcrafted fallback"
                logger.warning(
                    "[Re-ID] torchreid CUDA extraction failed; disabling torchreid for the rest of the run and using handcrafted fallback: %s",
                    exc,
                )
                if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                return self._extract_handcrafted(image)
            logger.warning("[Re-ID] torchreid extraction failed, falling back to handcrafted: %s", exc)
            return self._extract_handcrafted(image)

    def _extract_handcrafted(self, image: np.ndarray) -> Optional[np.ndarray]:
        gray = np.mean(image.astype(np.float32), axis=2)

        hist_b, _ = np.histogram(image[:, :, 0], bins=16, range=(0, 256))
        hist_g, _ = np.histogram(image[:, :, 1], bins=8, range=(0, 256))
        hist_r, _ = np.histogram(image[:, :, 2], bins=8, range=(0, 256))

        gy, gx = np.gradient(gray)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.mod(np.arctan2(gy, gx), 2 * np.pi)
        texture_hist, _ = np.histogram(angle, bins=8, range=(0, 2 * np.pi), weights=magnitude)

        upper = image[0 : max(1, int(image.shape[0] * 0.45)), :, :]
        lower = image[max(0, int(image.shape[0] * 0.45)) :, :, :]
        upper_bg, _ = np.histogram(upper[:, :, :2], bins=64, range=(0, 256))
        lower_bg, _ = np.histogram(lower[:, :, :2], bins=64, range=(0, 256))

        features = np.concatenate(
            [
                hist_b.astype(np.float32),
                hist_g.astype(np.float32),
                hist_r.astype(np.float32),
                texture_hist.astype(np.float32),
                upper_bg.astype(np.float32) * 1.5,
                lower_bg.astype(np.float32),
            ]
        ).astype(np.float32)
        return features

    def compute_similarity(self, emb1: Optional[np.ndarray], emb2: Optional[np.ndarray]) -> float:
        norm1 = _normalize_vector(emb1)
        norm2 = _normalize_vector(emb2)
        if norm1 is None or norm2 is None:
            return 0.0
        return float(np.clip(np.dot(norm1, norm2), -1.0, 1.0))

    def get_backend_status(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device": self.device,
            "torchreid_device": self.torchreid_device,
            "available_backends": {
                "fastreid": importlib.util.find_spec("fastreid") is not None,
                "torchreid": importlib.util.find_spec("torchreid") is not None,
                "torch": torch is not None,
                "cv2": cv2 is not None,
            },
            "resolved_config_path": str(self.resolved_config_path) if self.resolved_config_path else "",
            "resolved_weights_path": str(self.resolved_weights_path) if self.resolved_weights_path else "",
            "fallback_reason": self.backend_reason,
            "strict_fastreid": self.strict_fastreid,
            "backend_attempts": list(self.backend_history),
        }
