"""Shared analysis service used by both API endpoints and CLI."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from app.config.pipeline_config import OUTPUT_DIR, PipelineConfig
from app.event_detection.pipeline import MatchAnalysisPipeline


_SAFE_OUTPUT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass
class AnalysisRequestOptions:
    model: str = "yolov8n.pt"
    confidence: float = 0.3
    device: str = "auto"
    frame_skip: int = 1
    max_width: int = 1280
    max_height: int = 720
    enable_stabilization: bool = True
    enable_reid: bool = True
    enable_events: bool = True
    enable_minimap: bool = True
    enable_freeze_frames: bool = True
    quiet: bool = False
    enable_ml_detector: bool = False
    ml_model_path: Optional[str] = None
    ml_confidence: float = 0.7
    ml_device: str = "auto"


class AnalysisService:
    """Orchestrates pipeline config creation and execution."""

    def __init__(self, base_output_dir: Optional[Path] = None):
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path(OUTPUT_DIR)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_input_path(self, input_path: str | Path) -> Path:
        path = Path(input_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        else:
            path = path.resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Input video not found: {path}")
        return path

    def validate_ml_model_path(self, options: AnalysisRequestOptions) -> Optional[Path]:
        if not options.enable_ml_detector:
            return None
        if not options.ml_model_path:
            raise ValueError("ml_model_path is required when enable_ml_detector is true")
        model_path = Path(options.ml_model_path)
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
        else:
            model_path = model_path.resolve()
        if not model_path.exists() or not model_path.is_file():
            raise FileNotFoundError(f"ML model file not found: {model_path}")
        return model_path

    def sanitize_output_name(self, output_name: Optional[str], fallback_stem: str) -> str:
        if not output_name:
            return f"analysis_{fallback_stem}"
        if not _SAFE_OUTPUT_RE.match(output_name):
            raise ValueError("output_name may only contain letters, numbers, '.', '_' and '-'")
        cleaned = Path(output_name).stem
        if not cleaned:
            raise ValueError("output_name must contain at least one valid character")
        return cleaned

    def build_pipeline_config(
        self,
        options: AnalysisRequestOptions,
        input_video: Path,
        output_video: Path,
        output_json: Path,
    ) -> PipelineConfig:
        config = PipelineConfig()
        config.input_video = str(input_video)
        config.output_video = str(output_video)
        config.output_json = str(output_json)

        config.detection.model_name = options.model
        config.detection.confidence_threshold = options.confidence
        config.detection.device = options.device
        config.optimization.frame_skip = options.frame_skip
        config.optimization.max_input_width = options.max_width
        config.optimization.max_input_height = options.max_height

        config.preprocessing.enable_stabilization = options.enable_stabilization
        config.reid.enable = options.enable_reid
        config.event_detection.enable_ml_events = options.enable_events
        config.visualization.draw_minimap = options.enable_minimap
        config.event_detection.enable_freeze_frames = options.enable_freeze_frames
        config.verbose = not options.quiet

        config.ml_model.enable_ml_detector = options.enable_ml_detector
        config.ml_model.weights_path = options.ml_model_path or ""
        config.ml_model.ml_confidence_threshold = options.ml_confidence
        config.ml_model.ml_device = options.ml_device
        return config

    def create_output_paths(
        self,
        output_dir: Path,
        input_video: Path,
        output_name: Optional[str] = None,
    ) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = self.sanitize_output_name(output_name, input_video.stem)
        output_video = output_dir / f"{base_name}.mp4"
        output_json = output_dir / f"{base_name}.json"
        return output_video, output_json

    def run_analysis_with_paths(
        self,
        input_video: Path,
        output_video: Path,
        output_json: Path,
        options: AnalysisRequestOptions,
    ) -> Dict[str, Any]:
        self.validate_ml_model_path(options)
        config = self.build_pipeline_config(options, input_video, output_video, output_json)

        pipeline = MatchAnalysisPipeline(config)
        pipeline.initialize()
        result = pipeline.process_video(
            str(input_video),
            str(output_video),
            str(output_json),
        )

        artifact_paths = {
            "video": str(Path(result["output_video"]).resolve()),
            "json": str(Path(result["json_report"]).resolve()),
            "events_json": str((output_json.parent / "events.json").resolve()),
            "trajectories_csv": str((output_json.parent / "trajectories.csv").resolve()),
            "freeze_frames_json": str((output_json.parent / "freeze_frames.json").resolve()),
            "statsbomb_json": str(
                Path(result.get("statsbomb_report", output_json.parent / "statsbomb_events.json")).resolve()
            ),
        }
        self._write_freeze_frames_json(
            Path(artifact_paths["json"]),
            Path(artifact_paths["freeze_frames_json"]),
        )

        return {
            "output_video": result["output_video"],
            "json_report": result["json_report"],
            "frames_processed": int(result["frames_processed"]),
            "processing_time": float(result["processing_time"]),
            "avg_fps": float(result["avg_fps"]),
            "events_detected": int(result["events_detected"]),
            "event_summary": {k: int(v) for k, v in result.get("event_summary", {}).items()},
            "possession": [float(v) for v in result.get("possession", [50.0, 50.0])],
            "ml_detector_used": bool(result.get("ml_detector_used", False)),
            "artifact_paths": artifact_paths,
        }

    def run_job(
        self,
        job_id: str,
        input_path: str | Path,
        options: AnalysisRequestOptions,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        resolved_input = self.resolve_input_path(input_path)
        output_dir = self.base_output_dir / "api_jobs" / job_id
        output_video, output_json = self.create_output_paths(output_dir, resolved_input, output_name)
        result = self.run_analysis_with_paths(resolved_input, output_video, output_json, options)
        result["job_output_dir"] = str(output_dir.resolve())
        return result

    def _write_freeze_frames_json(self, report_path: Path, freeze_path: Path) -> None:
        if not report_path.exists():
            return
        try:
            with report_path.open("r", encoding="utf-8") as infile:
                payload = json.load(infile)
            events = payload.get("events", [])
            freeze_payload = []
            for event in events:
                freeze_frame = event.get("freeze_frame")
                if not freeze_frame:
                    continue
                freeze_payload.append(
                    {
                        "event_type": event.get("type"),
                        "frame": event.get("frame"),
                        "timestamp": event.get("timestamp"),
                        "player_id": event.get("player_id"),
                        "team_id": event.get("team_id"),
                        "position": event.get("position"),
                        "freeze_frame": freeze_frame,
                    }
                )
            with freeze_path.open("w", encoding="utf-8") as outfile:
                json.dump(freeze_payload, outfile, indent=2)
        except Exception:
            # Freeze-frame side export should never break the main job result.
            return
