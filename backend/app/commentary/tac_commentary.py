from app.commentary._root_module_loader import load_root_module as _load_root_module
import sys as _sys

_module = _load_root_module("tac_commentary.py", "tac_commentary")
_sys.modules[__name__] = _module
