from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fastapi_app.main import app
from fastapi_app.services.dependencies import job_service


@pytest.fixture(autouse=True)
def reset_job_state(tmp_path: Path):
    with job_service._lock:  # noqa: SLF001 - test-only reset
        job_service._jobs.clear()  # noqa: SLF001 - test-only reset
    job_service.analysis_service.base_output_dir = tmp_path
    yield
    with job_service._lock:  # noqa: SLF001 - test-only reset
        job_service._jobs.clear()  # noqa: SLF001 - test-only reset


@pytest.fixture
def client():
    return TestClient(app)


def _mark_job_completed(job_id: str, base_dir: Path):
    job_dir = base_dir / "api_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    video = job_dir / "analysis_test.mp4"
    video.write_bytes(b"video")
    report = job_dir / "analysis_test.json"
    report.write_text(
        '{"events":[{"type":"pass","frame":1}],"statistics":{"event_counts":{"pass":1}}}',
        encoding="utf-8",
    )
    events_json = job_dir / "events.json"
    events_json.write_text(
        '{"metadata":{"total_events":1},"event_summary":{"pass":1},"events":[{"type":"pass","frame":1}]}',
        encoding="utf-8",
    )
    traj_csv = job_dir / "trajectories.csv"
    traj_csv.write_text("player_id\n1\n", encoding="utf-8")
    freeze_json = job_dir / "freeze_frames.json"
    freeze_json.write_text("[]", encoding="utf-8")
    statsbomb_json = job_dir / "statsbomb_events.json"
    statsbomb_json.write_text('{"events":[]}', encoding="utf-8")

    with job_service._lock:  # noqa: SLF001 - test-only write
        record = job_service._jobs[job_id]  # noqa: SLF001 - test-only write
        record.status = "completed"
        record.started_at = "2026-01-01T00:00:01Z"
        record.completed_at = "2026-01-01T00:00:02Z"
        record.result = {
            "output_video": str(video),
            "json_report": str(report),
            "frames_processed": 10,
            "processing_time": 1.2,
            "avg_fps": 8.3,
            "events_detected": 1,
            "event_summary": {"pass": 1},
            "possession": [52.0, 48.0],
            "ml_detector_used": False,
            "job_output_dir": str(job_dir),
        }
        record.artifact_paths = {
            "video": str(video),
            "json": str(report),
            "events_json": str(events_json),
            "trajectories_csv": str(traj_csv),
            "freeze_frames_json": str(freeze_json),
            "statsbomb_json": str(statsbomb_json),
        }


def test_analyze_path_returns_job_id(client, monkeypatch, tmp_path: Path):
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"demo")

    def fake_start_job(job_id, input_path, options, output_name=None):
        _mark_job_completed(job_id, tmp_path)

    monkeypatch.setattr(job_service, "start_job", fake_start_job)

    response = client.post(
        "/api/v1/video/analyze/path",
        json={"input_path": str(input_video)},
    )
    assert response.status_code == 202
    payload = response.json()
    assert "job_id" in payload
    assert payload["status"] == "queued"


def test_upload_submission_and_result_flow(client, monkeypatch, tmp_path: Path):
    def fake_start_job(job_id, input_path, options, output_name=None):
        _mark_job_completed(job_id, tmp_path)

    monkeypatch.setattr(job_service, "start_job", fake_start_job)

    response = client.post(
        "/api/v1/video/analyze/upload",
        files={"file": ("upload.mp4", b"binary", "video/mp4")},
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]

    status_response = client.get(f"/api/v1/video/jobs/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    result_response = client.get(f"/api/v1/video/jobs/{job_id}/result")
    assert result_response.status_code == 200
    assert result_response.json()["events_detected"] == 1

    events_response = client.get(f"/api/v1/video/jobs/{job_id}/events")
    assert events_response.status_code == 200
    assert events_response.json()["event_summary"]["pass"] == 1

    artifact_response = client.get(f"/api/v1/video/jobs/{job_id}/artifacts/json")
    assert artifact_response.status_code == 200


def test_invalid_input_path_returns_404(client):
    response = client.post(
        "/api/v1/video/analyze/path",
        json={"input_path": "does-not-exist.mp4"},
    )
    assert response.status_code == 404


def test_missing_ml_model_returns_404(client, tmp_path: Path):
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"demo")
    response = client.post(
        "/api/v1/video/analyze/path",
        json={
            "input_path": str(input_video),
            "enable_ml_detector": True,
            "ml_model_path": str(tmp_path / "missing_model.pth"),
        },
    )
    assert response.status_code == 404
