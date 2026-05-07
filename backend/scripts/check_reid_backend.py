"""Windows-friendly diagnostic for the backend Re-ID runtime."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def main() -> int:
    backend_root = Path(__file__).resolve().parents[1]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

    from app.config.pipeline_config import PipelineConfig
    from app.event_detection.reid_module import ReIDModel
    from app.utils.runtime_compat import apply_runtime_compatibility_shims

    apply_runtime_compatibility_shims()

    config = PipelineConfig()
    model = ReIDModel(config=config.reid)
    status = model.get_backend_status()

    diagnostics = {
        "python_executable": sys.executable,
        "backend_root": str(backend_root),
        "active_venv_matches_backend": "backend\\venv" in sys.executable.lower() or ".venv" in sys.executable.lower(),
        "imports": {
            "fastreid": importlib.util.find_spec("fastreid") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
            "torchvision": importlib.util.find_spec("torchvision") is not None,
            "cv2": importlib.util.find_spec("cv2") is not None,
            "torchreid": importlib.util.find_spec("torchreid") is not None,
        },
        "python312_collections_shim": True,
        "torch_six_shim": True,
        "paths": {
            "fastreid_config_path": status["resolved_config_path"],
            "fastreid_config_exists": bool(status["resolved_config_path"]) and Path(status["resolved_config_path"]).exists(),
            "fastreid_weights_path": status["resolved_weights_path"],
            "fastreid_weights_exists": bool(status["resolved_weights_path"]) and Path(status["resolved_weights_path"]).exists(),
        },
        "selected_backend": status["backend"],
        "fallback_reason": status["fallback_reason"],
        "strict_fastreid": status["strict_fastreid"],
        "backend_attempts": status["backend_attempts"],
    }

    print(json.dumps(diagnostics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
