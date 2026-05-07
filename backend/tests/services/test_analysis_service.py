from pathlib import Path

from app.services.analysis_service import AnalysisRequestOptions, AnalysisService


def test_build_pipeline_config_matches_expected_defaults(tmp_path: Path):
    service = AnalysisService(base_output_dir=tmp_path)
    options = AnalysisRequestOptions()
    config = service.build_pipeline_config(
        options=options,
        input_video=tmp_path / "input.mp4",
        output_video=tmp_path / "out.mp4",
        output_json=tmp_path / "out.json",
    )

    assert config.detection.model_name == "yolov8n.pt"
    assert config.detection.confidence_threshold == 0.3
    assert config.detection.device == "auto"
    assert config.optimization.frame_skip == 1
    assert config.optimization.max_input_width == 1280
    assert config.optimization.max_input_height == 720
    assert config.preprocessing.enable_stabilization is True
    assert config.reid.enable is True
    assert config.event_detection.enable_ml_events is True
    assert config.visualization.draw_minimap is True
    assert config.event_detection.enable_freeze_frames is True
    assert config.ml_model.enable_ml_detector is False
    assert config.verbose is True

