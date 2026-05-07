"""Lightweight football-content validation for uploaded match videos."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config.settings import settings

_YOLO_VALIDATION_MODEL: Any | None = None
_YOLO_VALIDATION_UNAVAILABLE = False


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


def _get_validation_yolo_model() -> Any | None:
    """Load a cached COCO detector for semantic preflight evidence when available."""
    global _YOLO_VALIDATION_MODEL, _YOLO_VALIDATION_UNAVAILABLE
    if _YOLO_VALIDATION_MODEL is not None:
        return _YOLO_VALIDATION_MODEL
    if _YOLO_VALIDATION_UNAVAILABLE:
        return None

    try:
        from ultralytics import YOLO

        model_path = Path(__file__).resolve().parents[2] / "yolov8n.pt"
        if not model_path.exists():
            model_path = Path("yolov8n.pt")
        _YOLO_VALIDATION_MODEL = YOLO(str(model_path))
        return _YOLO_VALIDATION_MODEL
    except Exception:
        _YOLO_VALIDATION_UNAVAILABLE = True
        return None


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


def _cricket_crease_score(frame: np.ndarray, hsv: np.ndarray, green_mask: np.ndarray) -> float:
    """Detect cricket crease markings: short parallel white lines on wicket grass."""
    h, w = frame.shape[:2]
    if np.count_nonzero(green_mask) < 0.10 * green_mask.size:
        return 0.0

    white_mask = cv2.inRange(hsv, np.array([0, 0, 145], dtype=np.uint8), np.array([180, 90, 255], dtype=np.uint8))
    white_mask[: int(h * 0.08), :] = 0
    edges = cv2.Canny(white_mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=max(18, int(min(h, w) * 0.045)),
        minLineLength=max(22, int(w * 0.07)),
        maxLineGap=max(5, int(w * 0.025)),
    )
    if lines is None:
        return 0.0

    horizontal_segments: list[tuple[float, float, float]] = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < w * 0.07:
            continue
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 12 and angle < 168:
            continue
        center_y = (y1 + y2) / (2 * h)
        if not (0.16 <= center_y <= 0.88):
            continue
        horizontal_segments.append((center_y, min(x1, x2) / w, max(x1, x2) / w))

    if not horizontal_segments:
        return 0.0

    y_clusters: list[float] = []
    for center_y, _x1, _x2 in sorted(horizontal_segments):
        if not y_clusters or abs(center_y - y_clusters[-1]) > 0.045:
            y_clusters.append(center_y)

    segment_score = _clamp(len(horizontal_segments) / 4.0)
    cluster_score = _clamp(len(y_clusters) / 2.0)
    return _clamp(0.55 * segment_score + 0.45 * cluster_score)


def _wicket_stump_score(frame: np.ndarray, hsv: np.ndarray, green_mask: np.ndarray) -> float:
    """Detect narrow vertical wicket/stump-like objects near the batting crease."""
    h, w = frame.shape[:2]
    if np.count_nonzero(green_mask) < 0.10 * green_mask.size:
        return 0.0

    saturated = cv2.inRange(hsv, np.array([0, 65, 55], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    bright = cv2.inRange(hsv, np.array([0, 0, 150], dtype=np.uint8), np.array([180, 95, 255], dtype=np.uint8))
    non_green = cv2.bitwise_not(green_mask)
    candidate_mask = cv2.bitwise_or(saturated, bright)
    candidate_mask = cv2.bitwise_and(candidate_mask, non_green)
    candidate_mask[:, : int(w * 0.18)] = 0
    candidate_mask[:, int(w * 0.82) :] = 0
    candidate_mask[int(h * 0.88) :, :] = 0
    candidate_mask = cv2.morphologyEx(
        candidate_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5)),
    )

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = 0.0
    frame_area = h * w
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.00012 or area > frame_area * 0.035:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bh < h * 0.055 or bw > w * 0.085:
            continue
        aspect = bh / max(1, bw)
        center_x = (x + bw / 2) / w
        center_y = (y + bh / 2) / h
        if aspect < 1.8 or not (0.25 <= center_x <= 0.75) or center_y > 0.72:
            continue
        height_score = _clamp(bh / (h * 0.22))
        aspect_score = _clamp((aspect - 1.8) / 4.0)
        best = max(best, _clamp(0.60 * height_score + 0.40 * aspect_score))
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


def _semantic_object_score(frame: np.ndarray, green_mask: np.ndarray) -> dict[str, float]:
    """Use COCO object detections as optional evidence for football-like gameplay."""
    model = _get_validation_yolo_model()
    if model is None:
        return {
            "semantic_available": 0.0,
            "person_count": 0.0,
            "person_on_pitch": 0.0,
            "player_distribution": 0.0,
            "small_player_ratio": 0.0,
            "sports_ball": 0.0,
            "non_football_equipment": 0.0,
            "closeup_person": 0.0,
            "semantic_football": 0.0,
        }

    h, w = frame.shape[:2]
    frame_area = h * w
    green_dilated = cv2.dilate(
        green_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)),
        iterations=1,
    )

    try:
        results = model(frame, conf=0.20, iou=0.45, imgsz=640, verbose=False)[0]
    except Exception:
        return {
            "semantic_available": 0.0,
            "person_count": 0.0,
            "person_on_pitch": 0.0,
            "player_distribution": 0.0,
            "small_player_ratio": 0.0,
            "sports_ball": 0.0,
            "non_football_equipment": 0.0,
            "closeup_person": 0.0,
            "semantic_football": 0.0,
        }

    person_centers: list[tuple[float, float]] = []
    small_persons = 0
    persons_on_pitch = 0
    balls = 0
    non_football_equipment = 0
    closeup_person = 0.0
    equipment_class_ids = {
        29,  # frisbee
        30,  # skis
        31,  # snowboard
        36,  # skateboard
        37,  # surfboard
        38,  # tennis racket
        39,  # baseball bat, often catches cricket bats
        40,  # baseball glove
    }

    boxes = getattr(results, "boxes", None)
    if boxes is not None:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.20:
                continue
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].cpu().numpy()]
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            area_ratio = (bw * bh) / frame_area
            cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
            foot_y = int(np.clip(y2, 0, h - 1))
            center = (cx / w, ((y1 + y2) / 2) / h)

            if cls_id == 0:
                person_centers.append(center)
                if green_dilated[foot_y, cx] > 0:
                    persons_on_pitch += 1
                if 0.00025 <= area_ratio <= 0.045:
                    small_persons += 1
                if area_ratio >= 0.18:
                    closeup_person = max(closeup_person, _clamp((area_ratio - 0.18) / 0.30))
            elif cls_id == 32 and conf >= 0.22:
                balls += 1
            elif cls_id in equipment_class_ids and conf >= 0.28:
                non_football_equipment += 1

    person_count_score = _clamp(len(person_centers) / 10.0)
    person_on_pitch_score = _clamp(persons_on_pitch / 8.0)
    small_player_score = _clamp(small_persons / 8.0)
    if len(person_centers) >= 3:
        xs = np.array([x for x, _y in person_centers], dtype=np.float32)
        ys = np.array([y for _x, y in person_centers], dtype=np.float32)
        spread = _clamp((float(np.std(xs)) + float(np.std(ys))) / 0.34)
    else:
        spread = 0.0

    semantic_football = _clamp(
        0.28 * person_count_score
        + 0.30 * person_on_pitch_score
        + 0.22 * small_player_score
        + 0.14 * spread
        + 0.06 * _clamp(balls / 2.0)
        - 0.25 * _clamp(non_football_equipment / 2.0)
        - 0.20 * closeup_person
    )

    return {
        "semantic_available": 1.0,
        "person_count": float(len(person_centers)),
        "person_on_pitch": person_on_pitch_score,
        "player_distribution": spread,
        "small_player_ratio": small_player_score,
        "sports_ball": _clamp(balls / 2.0),
        "non_football_equipment": _clamp(non_football_equipment / 2.0),
        "closeup_person": closeup_person,
        "semantic_football": semantic_football,
    }


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
    cricket_crease = _cricket_crease_score(frame, hsv, green_mask)
    wicket_stumps = _wicket_stump_score(frame, hsv, green_mask)
    raw_cricket_specific = max(cricket_pitch, _clamp(0.55 * cricket_crease + 0.45 * wicket_stumps))
    cricket_specific = raw_cricket_specific
    if football_markings >= 0.50 and cricket_pitch < 0.20 and wicket_stumps < 0.70:
        cricket_specific *= 0.15
    players = _player_blob_score(frame, green_mask)
    broadcast = _broadcast_overlay_score(frame, green_mask)
    semantic = _semantic_object_score(frame, green_mask)

    score = _clamp(
        0.45 * field_score
        + 0.23 * football_markings
        + 0.14 * line
        + 0.08 * players
        + 0.15 * semantic["semantic_football"]
        + 0.04 * broadcast
        - 0.65 * cricket_specific
        - 0.15 * semantic["non_football_equipment"]
        - 0.08 * semantic["closeup_person"]
    )
    return {
        "score": score,
        "field": field_score,
        "green_ratio": _clamp(green_ratio),
        "largest_green_ratio": _clamp(largest_green_ratio),
        "line": line,
        "football_markings": football_markings,
        "cricket_pitch": cricket_pitch,
        "cricket_crease": cricket_crease,
        "wicket_stumps": wicket_stumps,
        "cricket_specific": cricket_specific,
        "player_blobs": players,
        "broadcast_overlay": broadcast,
        **semantic,
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
        "cricket_crease": float(np.mean([item["cricket_crease"] for item in frame_scores])),
        "wicket_stumps": float(np.mean([item["wicket_stumps"] for item in frame_scores])),
        "cricket_specific": float(np.mean([item["cricket_specific"] for item in frame_scores])),
        "player_blobs": float(np.mean([item["player_blobs"] for item in frame_scores])),
        "broadcast_overlay": float(np.mean([item["broadcast_overlay"] for item in frame_scores])),
        "semantic_available": float(np.mean([item["semantic_available"] for item in frame_scores])),
        "person_count": float(np.mean([item["person_count"] for item in frame_scores])),
        "person_on_pitch": float(np.mean([item["person_on_pitch"] for item in frame_scores])),
        "player_distribution": float(np.mean([item["player_distribution"] for item in frame_scores])),
        "small_player_ratio": float(np.mean([item["small_player_ratio"] for item in frame_scores])),
        "sports_ball": float(np.mean([item["sports_ball"] for item in frame_scores])),
        "non_football_equipment": float(np.mean([item["non_football_equipment"] for item in frame_scores])),
        "closeup_person": float(np.mean([item["closeup_person"] for item in frame_scores])),
        "semantic_football": float(np.mean([item["semantic_football"] for item in frame_scores])),
    }

    min_confidence = float(settings.FOOTBALL_VIDEO_VALIDATION_MIN_CONFIDENCE)
    uncertain_confidence = float(settings.FOOTBALL_VIDEO_VALIDATION_UNCERTAIN_CONFIDENCE)
    allow_uncertain = bool(settings.FOOTBALL_VIDEO_VALIDATION_ALLOW_UNCERTAIN)

    cricket_pitch = evidence["cricket_pitch"]
    cricket_specific = evidence["cricket_specific"]
    cricket_crease = evidence["cricket_crease"]
    wicket_stumps = evidence["wicket_stumps"]
    football_markings = evidence["football_markings"]
    football_line_evidence = max(football_markings, evidence["line"])
    semantic_available = evidence["semantic_available"] >= 0.50
    semantic_football = evidence["semantic_football"]
    non_football_equipment = evidence["non_football_equipment"]
    closeup_person = evidence["closeup_person"]
    has_football_scene = (
        football_line_evidence >= 0.24
        and football_markings >= 0.16
        and evidence["field"] >= 0.35
    )
    has_strong_pitch_without_yolo_players = (
        evidence["person_count"] < 1.0
        and evidence["field"] >= 0.60
        and football_markings >= 0.50
        and football_line_evidence >= 0.70
    )
    has_semantic_match_context = (
        not semantic_available
        or semantic_football >= 0.10
        or has_strong_pitch_without_yolo_players
    )

    if (
        cricket_pitch >= 0.32
        or cricket_specific >= 0.35
        or (cricket_crease >= 0.55 and wicket_stumps >= 0.70)
        or (cricket_crease >= 0.55 and football_markings < 0.45)
        or (semantic_available and non_football_equipment >= 0.25 and semantic_football < 0.35)
        or (semantic_available and closeup_person >= 0.45 and not has_football_scene)
    ):
        status = "invalid"
        is_valid = False
        message = (
            "The uploaded video is not a football video. "
            "Please upload football match footage showing a football pitch and gameplay."
        )
    elif (
        confidence >= min_confidence
        and positive_ratio >= 0.30
        and has_football_scene
        and has_semantic_match_context
    ):
        status = "accepted"
        is_valid = True
        message = "Football match video validated."
    elif (
        confidence >= uncertain_confidence
        and allow_uncertain
        and football_line_evidence >= 0.18
        and cricket_specific < 0.25
        and has_semantic_match_context
    ):
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
            "The uploaded video is not a football video. "
            "Please upload football match footage showing a football pitch and gameplay."
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
