import time
from pathlib import Path

from app.services.analysis_service import AnalysisRequestOptions
from app.services.job_service import JobService


class StubAnalysisService:
    def __init__(self, base_output_dir: Path):
        self.base_output_dir = base_output_dir

    def resolve_input_path(self, input_path):
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input video not found: {path}")
        return path.resolve()

    def validate_ml_model_path(self, options):
        if options.enable_ml_detector and not options.ml_model_path:
            raise ValueError("ml_model_path is required when enable_ml_detector is true")
        return None

    def sanitize_output_name(self, output_name, fallback_stem):
        return output_name or fallback_stem

    def run_job(self, job_id, input_path, options, output_name=None):
        input_path = Path(input_path)
        if "fail" in input_path.name:
            raise RuntimeError("forced failure")
        job_dir = self.base_output_dir / "api_jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        report = job_dir / "analysis.json"
        report.write_text('{"events":[],"statistics":{"event_counts":{}}}', encoding="utf-8")
        video = job_dir / "analysis.mp4"
        video.write_bytes(b"video")
        events = job_dir / "events.json"
        events.write_text(
            '{"metadata":{"total_events":0},"event_summary":{},"events":[]}',
            encoding="utf-8",
        )
        traj = job_dir / "trajectories.csv"
        traj.write_text("player_id\n", encoding="utf-8")
        freeze = job_dir / "freeze_frames.json"
        freeze.write_text("[]", encoding="utf-8")
        statsbomb = job_dir / "statsbomb_events.json"
        statsbomb.write_text('{"events":[]}', encoding="utf-8")
        return {
            "output_video": str(video),
            "json_report": str(report),
            "frames_processed": 1,
            "processing_time": 0.1,
            "avg_fps": 10.0,
            "events_detected": 0,
            "event_summary": {},
            "possession": [50.0, 50.0],
            "ml_detector_used": False,
            "job_output_dir": str(job_dir),
            "artifact_paths": {
                "video": str(video),
                "json": str(report),
                "events_json": str(events),
                "trajectories_csv": str(traj),
                "freeze_frames_json": str(freeze),
                "statsbomb_json": str(statsbomb),
            },
        }


def _wait_until_terminal(job_service: JobService, job_id: str, timeout_sec: float = 5.0):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        record = job_service.get_job(job_id)
        if record.status in {"completed", "failed"}:
            return record
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not complete in time")


def test_job_lifecycle_success_and_failure(tmp_path: Path):
    input_ok = tmp_path / "ok.mp4"
    input_ok.write_bytes(b"ok")
    input_fail = tmp_path / "fail.mp4"
    input_fail.write_bytes(b"fail")

    service = StubAnalysisService(base_output_dir=tmp_path)
    job_service = JobService(analysis_service=service, max_workers=2)

    success_job = job_service.create_job()
    job_service.start_job(success_job.job_id, input_ok, AnalysisRequestOptions())
    success_record = _wait_until_terminal(job_service, success_job.job_id)
    assert success_record.status == "completed"
    assert success_record.result["frames_processed"] == 1
    assert "video" in success_record.artifact_paths

    failed_job = job_service.create_job()
    job_service.start_job(failed_job.job_id, input_fail, AnalysisRequestOptions())
    failed_record = _wait_until_terminal(job_service, failed_job.job_id)
    assert failed_record.status == "failed"
    assert "forced failure" in (failed_record.error or "")

    job_service.shutdown()
