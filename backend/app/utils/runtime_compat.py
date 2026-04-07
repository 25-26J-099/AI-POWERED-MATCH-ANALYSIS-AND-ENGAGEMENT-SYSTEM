"""Runtime compatibility helpers for third-party libraries."""
from __future__ import annotations

import collections
import collections.abc
import sys
import types


def apply_python312_collection_aliases() -> None:
    """Restore deprecated ``collections`` aliases needed by legacy deps."""
    alias_names = (
        "Mapping",
        "MutableMapping",
        "Sequence",
        "MutableSequence",
        "Iterable",
        "Set",
        "MutableSet",
        "Callable",
    )
    for name in alias_names:
        if not hasattr(collections, name) and hasattr(collections.abc, name):
            setattr(collections, name, getattr(collections.abc, name))


def apply_torch_six_compatibility() -> None:
    """Provide the removed ``torch._six`` module for legacy libraries."""
    if "torch._six" in sys.modules:
        return

    shim = types.ModuleType("torch._six")
    shim.PY3 = True
    shim.string_classes = (str, bytes)
    shim.int_classes = (int,)
    shim.container_abcs = collections.abc
    sys.modules["torch._six"] = shim


def apply_runtime_compatibility_shims() -> None:
    """Apply all compatibility shims needed by legacy ML dependencies."""
    apply_python312_collection_aliases()
    apply_torch_six_compatibility()
