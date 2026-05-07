from pathlib import Path

import cv2
import numpy as np

from app.config.settings import settings
from app.services.football_video_validator import validate_football_video


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (width, height),
    )
    for frame in frames:
        writer.write(frame)
    writer.release()


def _football_like_frame() -> np.ndarray:
    frame = np.full((240, 320, 3), (45, 135, 45), dtype=np.uint8)
    cv2.line(frame, (0, 120), (320, 120), (245, 245, 245), 3)
    cv2.circle(frame, (160, 120), 35, (245, 245, 245), 2)
    for x, color in ((70, (30, 90, 220)), (110, (220, 60, 40)), (190, (30, 90, 220)), (240, (220, 60, 40))):
        cv2.rectangle(frame, (x, 95), (x + 10, 130), color, -1)
    return frame


def test_validator_accepts_pitch_like_video(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_FOOTBALL_VIDEO_VALIDATION", True)
    monkeypatch.setattr(settings, "FOOTBALL_VIDEO_VALIDATION_ALLOW_UNCERTAIN", False)
    monkeypatch.setattr(settings, "FOOTBALL_VIDEO_VALIDATION_SAMPLE_FRAMES", 6)

    video_path = tmp_path / "football.mp4"
    _write_video(video_path, [_football_like_frame() for _ in range(12)])

    result = validate_football_video(video_path)

    assert result.is_valid
    assert result.status == "accepted"
    assert result.confidence >= settings.FOOTBALL_VIDEO_VALIDATION_MIN_CONFIDENCE


def test_validator_rejects_non_football_video(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_FOOTBALL_VIDEO_VALIDATION", True)
    monkeypatch.setattr(settings, "FOOTBALL_VIDEO_VALIDATION_ALLOW_UNCERTAIN", False)
    monkeypatch.setattr(settings, "FOOTBALL_VIDEO_VALIDATION_SAMPLE_FRAMES", 6)

    video_path = tmp_path / "not_football.mp4"
    frames = [np.full((240, 320, 3), (40, 40, 40), dtype=np.uint8) for _ in range(12)]
    _write_video(video_path, frames)

    result = validate_football_video(video_path)

    assert not result.is_valid
    assert result.status == "invalid"


def test_validator_rejects_cricket_like_video(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_FOOTBALL_VIDEO_VALIDATION", True)
    monkeypatch.setattr(settings, "FOOTBALL_VIDEO_VALIDATION_ALLOW_UNCERTAIN", False)
    monkeypatch.setattr(settings, "FOOTBALL_VIDEO_VALIDATION_SAMPLE_FRAMES", 6)

    frame = np.full((240, 320, 3), (45, 135, 45), dtype=np.uint8)
    cv2.rectangle(frame, (135, 35), (185, 210), (95, 150, 190), -1)
    cv2.line(frame, (125, 70), (195, 70), (245, 245, 245), 2)
    cv2.line(frame, (125, 175), (195, 175), (245, 245, 245), 2)
    for x, y in ((145, 65), (175, 175), (80, 105), (235, 130)):
        cv2.rectangle(frame, (x, y), (x + 9, y + 28), (220, 60, 40), -1)

    video_path = tmp_path / "cricket.mp4"
    _write_video(video_path, [frame.copy() for _ in range(12)])

    result = validate_football_video(video_path)

    assert not result.is_valid
    assert result.status == "invalid"
    assert result.evidence["cricket_pitch"] >= 0.32
