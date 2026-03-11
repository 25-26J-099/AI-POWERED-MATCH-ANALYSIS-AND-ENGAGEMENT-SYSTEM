"""
Geometry and mathematical utility functions for spatial analysis.
"""
import numpy as np
from typing import Tuple, Optional, List


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_velocity(
    positions: List[Tuple[float, float]],
    fps: float = 30.0
) -> float:
    """Calculate average velocity from a list of positions (pixels/second)."""
    if len(positions) < 2:
        return 0.0
    total_dist = sum(
        calculate_distance(positions[i], positions[i + 1])
        for i in range(len(positions) - 1)
    )
    time_elapsed = (len(positions) - 1) / fps
    return total_dist / time_elapsed if time_elapsed > 0 else 0.0


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get center point of bounding box (x1, y1, x2, y2)."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def bbox_bottom_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Get bottom-center point (feet position) of bounding box."""
    return ((bbox[0] + bbox[2]) / 2, bbox[3])


def bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    """Calculate bounding box area."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """Calculate Intersection over Union of two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def point_in_region(
    point: Tuple[float, float],
    region: Tuple[float, float, float, float]
) -> bool:
    """Check if point is within a rectangular region (x1, y1, x2, y2)."""
    return (region[0] <= point[0] <= region[2] and
            region[1] <= point[1] <= region[3])


def predict_position(
    positions: List[Tuple[float, float]],
    n_future: int = 1
) -> Optional[Tuple[float, float]]:
    """Predict future position using linear extrapolation from recent positions."""
    if len(positions) < 2:
        return positions[-1] if positions else None

    # Use last few positions for velocity estimation
    recent = positions[-min(5, len(positions)):]
    dx = (recent[-1][0] - recent[0][0]) / (len(recent) - 1)
    dy = (recent[-1][1] - recent[0][1]) / (len(recent) - 1)

    return (
        recent[-1][0] + dx * n_future,
        recent[-1][1] + dy * n_future,
    )


def angle_between_points(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float]
) -> float:
    """Calculate angle at p2 formed by p1-p2-p3 (in degrees)."""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))


def compute_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray
) -> Optional[np.ndarray]:
    """Compute homography matrix from source to destination points."""
    import cv2
    if len(src_points) < 4:
        return None
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H


def transform_point(
    point: Tuple[float, float],
    homography: np.ndarray
) -> Tuple[float, float]:
    """Transform a point using a homography matrix."""
    p = np.array([point[0], point[1], 1.0])
    transformed = homography @ p
    transformed /= transformed[2]
    return (float(transformed[0]), float(transformed[1]))