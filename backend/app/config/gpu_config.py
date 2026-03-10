"""GPU/VM deployment configuration — provider-agnostic interface."""

import torch
from typing import Optional
from app.config.settings import settings


def get_device() -> torch.device:
    """Return the best available compute device.

    Priority: CUDA GPU > MPS (Apple Silicon) > CPU
    Respects FORCE_CPU setting.
    """
    if settings.FORCE_CPU:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Return device info for diagnostics."""
    device = get_device()
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "gpu_provider": settings.GPU_PROVIDER,
        "force_cpu": settings.FORCE_CPU,
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / 1e9, 2)
    return info


# ── Provider Placeholder Configs ─────────────────────────────────────────

VASTAI_CONFIG = {
    "api_key": settings.VASTAI_API_KEY,
    "instance_type": "gpu",  # placeholder — set via env
    "gpu_type": "RTX_3090",  # placeholder
    "disk_gb": 40,
}

RUNPOD_CONFIG = {
    "api_key": settings.RUNPOD_API_KEY,
    "gpu_type": "NVIDIA A40",  # placeholder
    "cloud_type": "COMMUNITY",  # or "SECURE"
    "volume_gb": 40,
}


def get_provider_config() -> Optional[dict]:
    """Return the active provider config, or None for local."""
    if settings.GPU_PROVIDER == "vastai":
        return VASTAI_CONFIG
    elif settings.GPU_PROVIDER == "runpod":
        return RUNPOD_CONFIG
    return None
