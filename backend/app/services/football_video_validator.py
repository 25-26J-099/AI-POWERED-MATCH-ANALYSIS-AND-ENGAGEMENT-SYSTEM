"""Lightweight football-content validation for uploaded match videos."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config.settings import settings


@dataclass
class FootballVideoValidationResult:
    """Structured result returned by the football-video upload gate."""

    is_valid: bool
    status: str
    confidence: float
    message: str
    sampled_frames: int = 0
    positive_frame_ratio: float = 0.0
    evidence: dict[str, float] = field(default_factory=dict)
    frame_scores: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _resize_for_validation(frame: np.ndarray, max_side: int = 480) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _field_green_mask(hsv: np.ndarray) -> np.ndarray:
    """Return a broad pitch-green mask robust to amateur-camera lighting."""
    lower = np.array([30, 35, 35], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _largest_component_ratio(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(cv2.contourArea(contour) for contour in contours)
    return float(largest) / float(mask.shape[0] * mask.shape[1])


def _line_score(hsv: np.ndarray, green_mask: np.ndarray) -> float:
    """Estimate football pitch markings from white pixels near green regions."""
    white_mask = cv2.inRange(hsv, np.array([0, 0, 150], dtype=np.uint8), np.array([180, 80, 255], dtype=np.uint8))
    if green_mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        near_pitch = cv2.dilate(green_mask, kernel, iterations=1)
        white_mask = cv2.bitwise_and(white_mask, near_pitch)
    white_ratio = float(np.count_nonzero(white_mask)) / float(white_mask.size)
    return _clamp(white_ratio / 0.025)


def _football_marking_score(frame: np.ndarray, hsv: np.ndarray, green_mask: np.ndarray) -> float:
    """Estimate football-style pitch markings using long white lines on grass.

    Cricket clips can also contain green fields and players, so this signal looks
    for repeated long markings distributed on the grass rather than just any
    broadcast/scoreboard or bright object.
    """
    h, w = frame.shape[:2]
    white_mask = cv2.inRange(hsv, np.array([0, 0, 155], dtype=np.uint8), np.array([180, 75, 255], dtype=np.uint8))
    near_pitch = cv2.dilate(green_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1)
    white_mask = cv2.bitwise_and(white_mask, near_pitch)
    white_mask[: int(h * 0.10), :] = 0

    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(20, int(min(h, w) * 0.06)),
        minLineLength=max(24, int(min(h, w) * 0.16)),
        maxLineGap=max(6, int(min(h, w) * 0.035)),
    )
    if lines is None:
        return 0.0

    long_lines = 0
    orientation_bins: set[str] = set()
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min(h, w) * 0.16:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        long_lines += 1
        if angle < 20 or angle > 160:
            orientation_bins.add("horizontal")
        elif 70 < angle < 110:
            orientation_bins.add("vertical")
        else:
            orientation_bins.add("diagonal")

    line_component = _clamp(long_lines / 5.0)
    diversity_component = _clamp(len(orientation_bins) / 2.0)
    return _clamp(0.70 * line_component + 0.30 * diversity_component)


def _cricket_pitch_score(frame: np.ndarray, hsv: np.ndarray, green_mask: np.ndarray) -> float:
    """Detect a cricket wicket/pitch: elongated tan/brown strip inside green field."""
    h, w = frame.shape[:2]
    if np.count_nonzero(green_mask) < 0.12 * green_mask.size:
        return 0.0

    tan_mask = cv2.inRange(hsv, np.array([5, 25, 65], dtype=np.uint8), np.array([35, 190, 235], dtype=np.uint8))
    # Ignore common broadcast overlay area and require the strip to be embedded in/near grass.
    tan_mask[: int(h * 0.10), :] = 0
    near_pitch = cv2.dilate(green_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=1)
    tan_mask = cv2.bitwise_and(tan_mask, near_pitch)
    tan_mask = cv2.morphologyEx(tan_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    tan_mask = cv2.morphologyEx(tan_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13)))

    contours, _ = cv2.findContours(tan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    frame_area = h * w
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.004 or area > frame_area * 0.20:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = max(bw, bh) / max(1, min(bw, bh))
        center_y = (y + bh / 2) / h
        if aspect < 2.0 or not (0.18 <= center_y <= 0.88):
            continue
        area_score = _clamp(area / (frame_area * 0.045))
        aspect_score = _clamp((aspect - 2.0) / 4.0)
        best = max(best, _clamp(0.65 * area_score + 0.35 * aspect_score))
    return best


def _player_blob_score(frame: np.ndarray, green_mask: np.ndarray) -> float:
    """Estimate player-like foreground blobs over the pitch without loading YOLO."""
    h, w = frame.shape[:2]
    pitch_area = np.count_nonzero(green_mask)
    if pitch_area < 0.08 * green_mask.size:
        return 0.0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    pitch_region = cv2.dilate(green_mask, kernel, iterations=1)
    non_green_on_pitch = cv2.bitwise_and(cv2.bitwise_not(green_mask), pitch_region)
    non_green_on_pitch[: int(h * 0.05), :] = 0

    contours, _ = cv2.findContours(non_green_on_pitch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plausible = 0
    frame_area = h * w
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.00008 or area > frame_area * 0.02:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        if 0.15 <= aspect <= 1.8 and bh >= max(8, h * 0.025):
            plausible += 1
    return _clamp(plausible / 8.0)


def _broadcast_overlay_score(frame: np.ndarray, green_mask: np.ndarray) -> float:
    """Weak evidence for match footage: scoreboard/banner-like overlays in the top band."""
    h, w = frame.shape[:2]
    top = frame[: max(1, int(h * 0.18)), :]
    top_green = green_mask[: top.shape[0], :]
    if np.count_nonzero(top_green) / max(top_green.size, 1) > 0.45:
        return 0.0

    gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw >= 0.12 * w and 8 <= bh <= 0.16 * h:
            return 1.0
    return 0.0


def _score_frame(frame: np.ndarray) -> dict[str, float]:
    frame = _resize_for_validation(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = _field_green_mask(hsv)

    green_ratio = float(np.count_nonzero(green_mask)) / float(green_mask.size)
    largest_green_ratio = _largest_component_ratio(green_mask)
    field_score = _clamp((green_ratio - 0.08) / 0.32)
    field_score = max(field_score, _clamp((largest_green_ratio - 0.06) / 0.28))

    line = _line_score(hsv, green_mask)
    football_markings = _football_marking_score(frame, hsv, green_mask)
    cricket_pitch = _cricket_pitch_score(frame, hsv, green_mask)
    players = _player_blob_score(frame, green_mask)
    broadcast = _broadcast_overlay_score(frame, green_mask)

    score = _clamp(
        0.45 * field_score
        + 0.23 * football_markings
        + 0.14 * line
        + 0.16 * players
        + 0.04 * broadcast
        - 0.55 * cricket_pitch
    )
    return {
        "score": score,
        "field": field_score,
        "green_ratio": _clamp(green_ratio),
        "largest_green_ratio": _clamp(largest_green_ratio),
        "line": line,
        "football_markings": football_markings,
        "cricket_pitch": cricket_pitch,
        "player_blobs": players,
        "broadcast_overlay": broadcast,
    }


def _sample_indices(total_frames: int, sample_count: int) -> list[int]:
    if total_frames <= 0:
        return []
    sample_count = max(1, min(sample_count, total_frames))
    if sample_count == 1:
        return [total_frames // 2]
    # Skip the very first/last frame because uploads often contain title cards or fades.
    start = int(total_frames * 0.08)
    end = max(start + 1, int(total_frames * 0.92))
    return sorted({int(idx) for idx in np.linspace(start, end, sample_count)})


def validate_football_video(video_path: str | Path) -> FootballVideoValidationResult:
    """Validate that an uploaded video plausibly contains football match footage.

    The validator intentionally uses fast OpenCV scene evidence instead of a heavy
    video classifier at upload time. It looks for repeated pitch-green field
    regions, football-style pitch-line evidence, player-like blobs on the field,
    weak broadcast-overlay cues, and negative evidence such as a cricket wicket.
    """
    if not settings.ENABLE_FOOTBALL_VIDEO_VALIDATION:
        return FootballVideoValidationResult(
            is_valid=True,
            status="skipped",
            confidence=1.0,
            message="Football video validation is disabled.",
        )

    path = Path(video_path)
    if not path.exists() or not path.is_file():
        return FootballVideoValidationResult(
            is_valid=False,
            status="invalid",
            confidence=0.0,
            message="Uploaded video could not be found for validation.",
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return FootballVideoValidationResult(
            is_valid=False,
            status="invalid",
            confidence=0.0,
            message="Uploaded file is not a readable video.",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_count = max(3, int(settings.FOOTBALL_VIDEO_VALIDATION_SAMPLE_FRAMES))
    indices = _sample_indices(total_frames, sample_count)
    frame_scores: list[dict[str, float]] = []

    try:
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_scores.append(_score_frame(frame))
    finally:
        cap.release()

    if not frame_scores:
        return FootballVideoValidationResult(
            is_valid=False,
            status="invalid",
            confidence=0.0,
            message="Uploaded file is not a readable video.",
        )

    scores = np.array([item["score"] for item in frame_scores], dtype=np.float32)
    top_count = max(1, int(np.ceil(len(scores) * 0.6)))
    top_mean = float(np.mean(np.sort(scores)[-top_count:]))
    positive_ratio = float(np.mean(scores >= 0.48))
    confidence = _clamp(0.70 * top_mean + 0.30 * positive_ratio)

    evidence = {
        "field": float(np.mean([item["field"] for item in frame_scores])),
        "green_ratio": float(np.mean([item["green_ratio"] for item in frame_scores])),
        "line": float(np.mean([item["line"] for item in frame_scores])),
        "football_markings": float(np.mean([item["football_markings"] for item in frame_scores])),
        "cricket_pitch": float(np.mean([item["cricket_pitch"] for item in frame_scores])),
        "player_blobs": float(np.mean([item["player_blobs"] for item in frame_scores])),
        "broadcast_overlay": float(np.mean([item["broadcast_overlay"] for item in frame_scores])),
    }

    min_confidence = float(settings.FOOTBALL_VIDEO_VALIDATION_MIN_CONFIDENCE)
    uncertain_confidence = float(settings.FOOTBALL_VIDEO_VALIDATION_UNCERTAIN_CONFIDENCE)
    allow_uncertain = bool(settings.FOOTBALL_VIDEO_VALIDATION_ALLOW_UNCERTAIN)

    cricket_pitch = evidence["cricket_pitch"]
    football_markings = evidence["football_markings"]
    football_specific_evidence = max(football_markings, evidence["line"], evidence["player_blobs"])

    if cricket_pitch >= 0.32:
        status = "invalid"
        is_valid = False
        message = (
            "This upload appears to be cricket or another non-football field sport. "
            "Please upload football match footage showing a football pitch and gameplay."
        )
    elif (
        confidence >= min_confidence
        and positive_ratio >= 0.30
        and football_specific_evidence >= 0.18
    ):
        status = "accepted"
        is_valid = True
        message = "Football match video validated."
    elif confidence >= uncertain_confidence and allow_uncertain:
        status = "uncertain"
        is_valid = True
        message = (
            "Football match content was partially detected. The upload was accepted, "
            "but later team-color detection may still reject unclear footage."
        )
    else:
        status = "invalid"
        is_valid = False
        message = (
            "This upload does not appear to contain football match footage. "
            "Please upload a video showing a football pitch, players, or match gameplay."
        )

    return FootballVideoValidationResult(
        is_valid=is_valid,
        status=status,
        confidence=round(confidence, 4),
        message=message,
        sampled_frames=len(frame_scores),
        positive_frame_ratio=round(positive_ratio, 4),
        evidence={key: round(value, 4) for key, value in evidence.items()},
        frame_scores=[{key: round(value, 4) for key, value in score.items()} for score in frame_scores],
    )
