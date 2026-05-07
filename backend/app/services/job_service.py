"""In-memory async job orchestration for video analysis requests."""
from __future__ import annotations

import copy
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Literal, Optional

from app.services.analysis_service import AnalysisRequestOptions, AnalysisService

JobStatus = Literal["queued", "running", "completed", "failed"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Dict[str, Any] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    progress: Optional[Dict[str, Any]] = None  # {frame, total, pct} updated every 100 frames


class JobService:
    """Thread-backed job queue with in-memory state tracking."""

    def __init__(self, analysis_service: AnalysisService, max_workers: int = 2):
        self.analysis_service = analysis_service
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = Lock()

    def create_job(self) -> JobRecord:
        job_id = str(uuid.uuid4())
        record = JobRecord(
            job_id=job_id,
            status="queued",
            submitted_at=_utc_now_iso(),
        )
        with self._lock:
            self._jobs[job_id] = record
        return copy.deepcopy(record)

    def get_job_output_dir(self, job_id: str) -> Path:
        return (self.analysis_service.base_output_dir / "api_jobs" / job_id).resolve()

    def get_job_input_path(self, job_id: str, filename: str) -> Path:
        safe_name = Path(filename).name or "uploaded_video.mp4"
        input_dir = self.get_job_output_dir(job_id) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        return (input_dir / safe_name).resolve()

    def start_job(
        self,
        job_id: str,
        input_path: str | Path,
        options: AnalysisRequestOptions,
        output_name: Optional[str] = None,
    ) -> None:
        resolved_input = self.analysis_service.resolve_input_path(input_path)
        self.analysis_service.validate_ml_model_path(options)
        if output_name:
            self.analysis_service.sanitize_output_name(output_name, "analysis")
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job id: {job_id}")
            if self._jobs[job_id].status != "queued":
                raise ValueError(f"Job {job_id} is already started")
        self.executor.submit(
            self._run_job,
            job_id,
            str(resolved_input),
            copy.deepcopy(options),
            output_name,
        )

    def update_job_progress(self, job_id: str, frame: int, total: int, pct: float) -> None:
        """Thread-safe update called from the pipeline thread every 100 frames."""
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].progress = {"frame": frame, "total": total, "pct": round(pct, 1)}

    def _run_job(
        self,
        job_id: str,
        input_path: str,
        options: AnalysisRequestOptions,
        output_name: Optional[str],
    ) -> None:
        with self._lock:
            record = self._jobs[job_id]
            record.status = "running"
            record.started_at = _utc_now_iso()
            record.error = None

        def _progress_cb(frame_idx: int, total: int, pct: float) -> None:
            self.update_job_progress(job_id, frame_idx, total, pct)

        try:
            result = self.analysis_service.run_job(
                job_id=job_id,
                input_path=input_path,
                options=options,
                output_name=output_name,
                progress_callback=_progress_cb,
            )
            artifact_paths = dict(result.pop("artifact_paths", {}))
            with self._lock:
                record = self._jobs[job_id]
                record.status = "completed"
                record.completed_at = _utc_now_iso()
                record.result = result
                record.artifact_paths = artifact_paths
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                record = self._jobs[job_id]
                record.status = "failed"
                record.completed_at = _utc_now_iso()
                record.error = str(exc)

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job id: {job_id}")
            return copy.deepcopy(self._jobs[job_id])

    def delete_job(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)

    def require_completed_job(self, job_id: str) -> JobRecord:
        record = self.get_job(job_id)
        if record.status != "completed":
            raise ValueError(f"Job {job_id} is not completed")
        return record

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
