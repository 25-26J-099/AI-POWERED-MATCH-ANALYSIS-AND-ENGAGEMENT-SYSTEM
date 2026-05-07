"""Bootstrap a FastReID-compatible ViT checkpoint for inference on Windows."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


TIMM_VIT_URL = (
    "https://github.com/rwightman/pytorch-image-models/releases/download/"
    "v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth"
)


def download_file(url: str, destination: Path) -> Path:
    import urllib.request

    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def main() -> int:
    from fastreid.config import get_cfg
    from fastreid.modeling.meta_arch import build_model

    config_path = BACKEND_ROOT / "models" / "reid" / "fastreid" / "configs" / "football_vit.yml"
    backbone_path = BACKEND_ROOT / "models" / "reid" / "fastreid" / "weights" / "jx_vit_base_p16_224-80ecf9dd.pth"
    checkpoint_path = BACKEND_ROOT / "models" / "reid" / "fastreid" / "weights" / "football_vit.pth"

    if not backbone_path.exists():
        download_file(TIMM_VIT_URL, backbone_path)

    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = ""

    model = build_model(cfg)
    model.eval()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, checkpoint_path)
    print(f"Saved FastReID-compatible checkpoint to {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
