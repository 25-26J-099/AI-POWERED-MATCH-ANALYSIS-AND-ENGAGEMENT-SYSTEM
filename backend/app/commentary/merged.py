from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from app.commentary._root_module_loader import load_root_module

_TAC = load_root_module("tac_commentary.py", "tac_commentary")
_PBP = load_root_module("pbp_commentary.py", "pbp_commentary")
_BACKEND_MERGED = load_root_module("merged.py", "merged")


def __getattr__(name: str):
    return getattr(_BACKEND_MERGED, name)


def __dir__():
    return sorted(set(globals()) | set(dir(_BACKEND_MERGED)))


def run_pipeline(
    events,
    threesixty_lookup,
    video_file,
    level,
    clip_filename=None,
    analytics_context=None,
    progress_callback=None,
    done_callback=None,
    audience_profile=None,
):
    """
    Compatibility adapter for the FastAPI service.

    The canonical commentary implementation lives in the repo-root `backend`
    scripts. The API currently calls a richer signature than the standalone
    script exposes, so this adapter materializes the passed event data into
    temporary JSON files and then delegates to the root `merged.py` pipeline.
    """
    del clip_filename, analytics_context, audience_profile

    temp_dir = Path(tempfile.mkdtemp(prefix="commentary_pipeline_"))
    events_path = temp_dir / "events.json"
    threesixty_path = temp_dir / "threesixty.json"

    with events_path.open("w", encoding="utf-8") as handle:
        json.dump(events, handle, ensure_ascii=False)
    with threesixty_path.open("w", encoding="utf-8") as handle:
        json.dump(list(threesixty_lookup.values()), handle, ensure_ascii=False)

    previous_paths = {
        "EVENT_FILE": getattr(_BACKEND_MERGED, "EVENT_FILE", None),
        "THREESIXTY_FILE": getattr(_BACKEND_MERGED, "THREESIXTY_FILE", None),
        "VIDEO_FILE": getattr(_BACKEND_MERGED, "VIDEO_FILE", None),
    }

    try:
        _BACKEND_MERGED.EVENT_FILE = str(events_path)
        _BACKEND_MERGED.THREESIXTY_FILE = str(threesixty_path)
        _BACKEND_MERGED.VIDEO_FILE = str(video_file)
        return _BACKEND_MERGED.run_pipeline(
            level,
            progress_callback=progress_callback,
            done_callback=done_callback,
        )
    finally:
        _BACKEND_MERGED.EVENT_FILE = previous_paths["EVENT_FILE"]
        _BACKEND_MERGED.THREESIXTY_FILE = previous_paths["THREESIXTY_FILE"]
        _BACKEND_MERGED.VIDEO_FILE = previous_paths["VIDEO_FILE"]
        shutil.rmtree(temp_dir, ignore_errors=True)
