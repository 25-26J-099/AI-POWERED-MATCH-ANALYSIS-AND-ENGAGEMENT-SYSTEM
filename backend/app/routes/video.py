"""Video analysis API endpoints."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from app.models.schemas import (
    AnalyzePathRequest, AnalysisResultResponse, EventListResponse, JobCreatedResponse, JobStatusResponse,
)
from app.services.analysis_service import AnalysisRequestOptions
from app.services.artifact_service import ArtifactService
from app.services.dependencies import get_artifact_service, get_job_service
from app.services.job_service import JobService

router = APIRouter()


def _to_options(payload: AnalyzePathRequest) -> AnalysisRequestOptions:
    return AnalysisRequestOptions(
        model=payload.model,
        confidence=payload.confidence,
        device=payload.device,
        frame_skip=payload.frame_skip,
        max_width=payload.max_width,
        max_height=payload.max_height,
        enable_stabilization=payload.enable_stabilization,
        enable_reid=payload.enable_reid,
        enable_events=payload.enable_events,
        enable_minimap=payload.enable_minimap,
        enable_freeze_frames=payload.enable_freeze_frames,
        quiet=payload.quiet,
        enable_ml_detector=payload.enable_ml_detector,
        ml_model_path=payload.ml_model_path,
        ml_confidence=payload.ml_confidence,
        ml_device=payload.ml_device,
    )


def _to_job_created_response(request: Request, job_id: str, submitted_at: str) -> JobCreatedResponse:
    return JobCreatedResponse(
        job_id=job_id,
        status="queued",
        submitted_at=submitted_at,
        status_url=str(request.url_for("get_job_status", job_id=job_id)),
        result_url=str(request.url_for("get_job_result", job_id=job_id)),
    )


@router.post(
    "/analyze/path",
    response_model=JobCreatedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def analyze_video_by_path(
    payload: AnalyzePathRequest,
    request: Request,
    job_service: JobService = Depends(get_job_service),
) -> JobCreatedResponse:
    options = _to_options(payload)
    record = job_service.create_job()
    try:
        job_service.start_job(
            job_id=record.job_id,
            input_path=payload.input_path,
            options=options,
            output_name=payload.output_name,
        )
    except FileNotFoundError as exc:
        job_service.delete_job(record.job_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        job_service.delete_job(record.job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_job_created_response(request, record.job_id, record.submitted_at)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, name="get_job_status")
def get_job_status(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
) -> JobStatusResponse:
    try:
        record = job_service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        submitted_at=record.submitted_at,
        started_at=record.started_at,
        completed_at=record.completed_at,
        error=record.error,
    )


@router.get("/jobs/{job_id}/result", response_model=AnalysisResultResponse, name="get_job_result")
def get_job_result(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
    artifact_service: ArtifactService = Depends(get_artifact_service),
) -> AnalysisResultResponse:
    try:
        record = job_service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if record.status == "failed":
        raise HTTPException(status_code=409, detail=record.error or "Job failed")
    if record.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job is {record.status}")

    result = record.result
    artifact_urls = artifact_service.build_artifact_urls(job_id)
    return AnalysisResultResponse(
        job_id=record.job_id,
        status=record.status,
        output_video=str(result["output_video"]),
        json_report=str(result["json_report"]),
        frames_processed=int(result["frames_processed"]),
        processing_time=float(result["processing_time"]),
        avg_fps=float(result["avg_fps"]),
        events_detected=int(result["events_detected"]),
        event_summary={k: int(v) for k, v in result.get("event_summary", {}).items()},
        possession=[float(v) for v in result.get("possession", [50.0, 50.0])],
        ml_detector_used=bool(result.get("ml_detector_used", False)),
        artifacts=artifact_urls,
    )


@router.get("/jobs/{job_id}/events", response_model=EventListResponse)
def get_job_events(
    job_id: str,
    job_service: JobService = Depends(get_job_service),
    artifact_service: ArtifactService = Depends(get_artifact_service),
) -> EventListResponse:
    try:
        record = job_service.require_completed_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        report_path = artifact_service.get_artifact_path(record, "json")
        events, summary = artifact_service.load_events(report_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return EventListResponse(job_id=job_id, event_summary=summary, events=events)


@router.get("/jobs/{job_id}/artifacts/{artifact_name}")
def download_artifact(
    job_id: str,
    artifact_name: str,
    job_service: JobService = Depends(get_job_service),
    artifact_service: ArtifactService = Depends(get_artifact_service),
) -> FileResponse:
    try:
        record = job_service.require_completed_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        artifact_path = artifact_service.get_artifact_path(record, artifact_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    return FileResponse(path=str(artifact_path), filename=Path(artifact_path).name)
