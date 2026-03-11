"""Artifact path validation and event extraction utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi_app.services.job_service import JobRecord


class ArtifactService:
    """Resolves and validates artifact files for completed jobs."""

    ALLOWED_ARTIFACTS = {
        "video",
        "json",
        "events_json",
        "trajectories_csv",
        "freeze_frames_json",
        "statsbomb_json",
    }

    def build_artifact_urls(self, job_id: str) -> Dict[str, str]:
        return {
            name: f"/api/v1/video/jobs/{job_id}/artifacts/{name}"
            for name in self.ALLOWED_ARTIFACTS
        }

    def get_artifact_path(self, record: JobRecord, artifact_name: str) -> Path:
        if artifact_name not in self.ALLOWED_ARTIFACTS:
            raise ValueError(f"Unsupported artifact: {artifact_name}")
        raw_path = record.artifact_paths.get(artifact_name)
        if not raw_path:
            raise FileNotFoundError(f"Artifact path not registered for '{artifact_name}'")
        path = Path(raw_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Artifact file not found: {path}")
        job_dir = Path(record.result.get("job_output_dir", "")).resolve()
        if job_dir and not path.is_relative_to(job_dir):
            raise PermissionError("Artifact path is outside the job directory")
        return path

    def load_events(self, report_path: Path) -> Tuple[List[dict], Dict[str, int]]:
        with report_path.open("r", encoding="utf-8") as infile:
            payload = json.load(infile)
        events = payload.get("events", [])
        summary = payload.get("statistics", {}).get("event_counts", {})
        if not summary:
            summary = {}
            for event in events:
                event_type = event.get("type", "unknown")
                summary[event_type] = int(summary.get(event_type, 0) + 1)
        return events, {k: int(v) for k, v in summary.items()}
