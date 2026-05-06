from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_BACKEND_DIR = Path(__file__).resolve().parents[2]


def load_root_module(root_filename: str, canonical_name: str) -> ModuleType:
    existing = sys.modules.get(canonical_name)
    if existing is not None:
        return existing

    module_path = _BACKEND_DIR / root_filename
    spec = importlib.util.spec_from_file_location(canonical_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load root commentary module: {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[canonical_name] = module
    spec.loader.exec_module(module)
    return module
