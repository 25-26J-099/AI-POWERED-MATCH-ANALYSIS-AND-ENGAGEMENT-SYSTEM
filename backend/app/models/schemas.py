"""Pydantic schemas for video analysis API endpoints."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


DeviceLiteral = Literal["auto", "cpu", "cuda"]
JobStatusLiteral = Literal["queued", "running", "completed", "failed"]


class AnalysisOptions(BaseModel):
    model: str = Field(default="yolov8n.pt")
    confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    device: DeviceLiteral = "auto"
    frame_skip: int = Field(default=1, ge=1)
    max_width: int = Field(default=1280, ge=1)
    max_height: int = Field(default=720, ge=1)

    enable_stabilization: bool = True
    enable_reid: bool = True
    enable_events: bool = True
    enable_minimap: bool = True
    enable_freeze_frames: bool = True
    quiet: bool = False

    enable_ml_detector: bool = False
    ml_model_path: Optional[str] = None
    ml_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    ml_device: DeviceLiteral = "auto"


class AnalyzePathRequest(AnalysisOptions):
    input_path: str = Field(..., min_length=1)
    output_name: Optional[str] = None


class AnalyzeUploadRequest(AnalysisOptions):
    output_name: Optional[str] = None


class JobCreatedResponse(BaseModel):
    job_id: str
    match_id: Optional[int] = None
    status: JobStatusLiteral
    submitted_at: str
    status_url: str
    result_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatusLiteral
    submitted_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class AnalysisResultResponse(BaseModel):
    job_id: str
    status: JobStatusLiteral
    output_video: str
    json_report: str
    frames_processed: int
    processing_time: float
    avg_fps: float
    events_detected: int
    event_summary: Dict[str, int]
    possession: List[float]
    ml_detector_used: bool
    artifacts: Dict[str, str]


class EventListResponse(BaseModel):
    job_id: str
    event_summary: Dict[str, int]
    events: List[Dict[str, Any]]
