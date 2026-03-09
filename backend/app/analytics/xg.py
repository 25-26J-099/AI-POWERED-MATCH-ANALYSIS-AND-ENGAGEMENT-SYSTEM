"""Expected Goals (xG) computation using pre-trained logistic regression model."""

import math
import numpy as np
from typing import Optional
from app.services.model_loader import load_xg_model


def _extract_features_from_event(event: dict) -> Optional[dict]:
    """Extract xG features from a shot event with optional freeze frame.

    Features:
        distance, angle, log_distance, distance_squared,
        angle_distance_interaction, pressure_weighted_distance,
        defender_count, nearest_defender_distance, goalkeeper_distance,
        has_360, body_part (categorical)
    """
    location = event.get("location")
    if not location or len(location) < 2:
        return None

    x, y = location[0], location[1]

    # Goal coordinates (StatsBomb: goal at x=120, y=36-44)
    goal_x, goal_y = 120.0, 40.0

    distance = math.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
    angle = math.atan2(abs(goal_y - y), abs(goal_x - x))

    # Extract freeze frame data
    freeze_frame = event.get("freeze_frame", [])
    has_360 = len(freeze_frame) > 0

    # Defaults for when no freeze frame
    defender_count = 0
    nearest_defender_distance = 30.0  # large default
    goalkeeper_distance = 30.0
    pressure_weighted_distance = distance

    if has_360:
        defenders = [p for p in freeze_frame if not p.get("teammate", True) and not p.get("keeper", False)]
        keepers = [p for p in freeze_frame if p.get("keeper", False) and not p.get("teammate", True)]

        defender_count = len(defenders)

        if defenders:
            def_dists = [
                math.sqrt((p["location"][0] - x) ** 2 + (p["location"][1] - y) ** 2)
                for p in defenders if "location" in p and len(p["location"]) >= 2
            ]
            if def_dists:
                nearest_defender_distance = min(def_dists)

        if keepers:
            gk = keepers[0]
            if "location" in gk and len(gk["location"]) >= 2:
                goalkeeper_distance = math.sqrt(
                    (gk["location"][0] - x) ** 2 + (gk["location"][1] - y) ** 2
                )

        # Pressure-weighted distance
        if nearest_defender_distance > 0:
            pressure_weighted_distance = distance * (1 + 1.0 / nearest_defender_distance)
        else:
            pressure_weighted_distance = distance * 2

    # Body part encoding
    shot_data = event.get("shot", {})
    body_part_raw = shot_data.get("body_part", {})
    if isinstance(body_part_raw, dict):
        body_part_name = body_part_raw.get("name", "Right Foot")
    else:
        body_part_name = str(body_part_raw) if body_part_raw else "Right Foot"

    body_part_map = {"Left Foot": 0, "Right Foot": 1, "Head": 2}
    body_part = body_part_map.get(body_part_name, 1)

    return {
        "distance": distance,
        "angle": angle,
        "log_distance": math.log(distance + 1),
        "distance_squared": distance ** 2,
        "angle_distance_interaction": angle * distance,
        "pressure_weighted_distance": pressure_weighted_distance,
        "defender_count": defender_count,
        "nearest_defender_distance": nearest_defender_distance,
        "goalkeeper_distance": goalkeeper_distance,
        "has_360": int(has_360),
        "body_part": body_part,
    }


def compute_xg(event: dict) -> float:
    """Compute xG for a single shot event.

    Uses the pre-trained logistic regression model loaded from HuggingFace.
    Falls back to a simple distance-based formula if model is unavailable.
    """
    features = _extract_features_from_event(event)
    if features is None:
        return 0.0

    feature_order = [
        "distance", "angle", "log_distance", "distance_squared",
        "angle_distance_interaction", "pressure_weighted_distance",
        "defender_count", "nearest_defender_distance", "goalkeeper_distance",
        "has_360", "body_part",
    ]
    X = np.array([[features[k] for k in feature_order]])

    model = load_xg_model()
    if model is not None:
        try:
            xg_value = float(model.predict_proba(X)[0][1])
            return min(max(xg_value, 0.0), 1.0)
        except Exception:
            pass

    # Fallback: simple distance-based xG
    distance = features["distance"]
    xg_value = max(0, 0.4 - 0.01 * distance)
    return min(max(xg_value, 0.0), 1.0)
