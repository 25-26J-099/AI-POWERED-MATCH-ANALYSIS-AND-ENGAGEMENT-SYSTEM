from pathlib import Path

import pytest

from fastapi_app.services.artifact_service import ArtifactService
from fastapi_app.services.job_service import JobRecord


def test_artifact_path_validation_and_safety(tmp_path: Path):
    artifact_service = ArtifactService()

    job_dir = tmp_path / "api_jobs" / "job-1"
    job_dir.mkdir(parents=True, exist_ok=True)
    video = job_dir / "analysis.mp4"
    video.write_bytes(b"video")
    report = job_dir / "analysis.json"
    report.write_text('{"events":[],"statistics":{"event_counts":{}}}', encoding="utf-8")

    outside = tmp_path / "outside.txt"
    outside.write_text("bad", encoding="utf-8")

    record = JobRecord(
        job_id="job-1",
        status="completed",
        submitted_at="2026-01-01T00:00:00Z",
        result={"job_output_dir": str(job_dir)},
        artifact_paths={
            "video": str(video),
            "json": str(report),
            "events_json": str(job_dir / "events.json"),
            "trajectories_csv": str(job_dir / "trajectories.csv"),
            "freeze_frames_json": str(job_dir / "freeze_frames.json"),
            "statsbomb_json": str(job_dir / "statsbomb_events.json"),
        },
    )

    assert artifact_service.get_artifact_path(record, "video") == video.resolve()

    with pytest.raises(ValueError):
        artifact_service.get_artifact_path(record, "not_allowed")

    record.artifact_paths["video"] = str(outside)
    with pytest.raises(PermissionError):
        artifact_service.get_artifact_path(record, "video")
