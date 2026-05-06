"""Expected Goals (xG) computation aligned with the training notebook."""
from __future__ import annotations

import math
import warnings
from typing import Optional

import pandas as pd

from app.services.model_loader import load_xg_model

GOAL_X_REAL = 105.0
GOAL_Y_REAL = 34.0
GOAL_WIDTH = 7.32


def _freeze_frame_players(event: dict) -> list[dict]:
    freeze_frame_raw = event.get("freeze_frame", [])
    if isinstance(freeze_frame_raw, dict):
        return freeze_frame_raw.get("players", [])
    return freeze_frame_raw if isinstance(freeze_frame_raw, list) else []


def _extract_features_from_event(event: dict) -> Optional[dict]:
    location = event.get("location")
    if not location or len(location) < 2:
        return None

    x, y = float(location[0]), float(location[1])
    x_real = x * GOAL_X_REAL / 120.0
    y_real = y * 68.0 / 80.0
    dx = GOAL_X_REAL - x_real
    dy = abs(GOAL_Y_REAL - y_real)
    distance = math.sqrt(dx**2 + dy**2)
    angle = abs(math.atan2(GOAL_WIDTH * dx, dx**2 + dy**2 - (GOAL_WIDTH / 2) ** 2))

    defender_count = 0
    nearest_defender_distance = 0.0
    goalkeeper_distance = 0.0

    freeze_frame = _freeze_frame_players(event)
    if freeze_frame:
        actor_location = [x, y]
        nearest_defender_distance = 999.0
        for player in freeze_frame:
            player_location = player.get("location")
            if not player_location or len(player_location) < 2:
                continue

            px, py = float(player_location[0]), float(player_location[1])
            distance_to_actor = math.dist([px, py], actor_location)

            if player.get("keeper", False):
                goalkeeper_distance = distance_to_actor

            if not player.get("teammate", True):
                defender_count += 1
                nearest_defender_distance = min(nearest_defender_distance, distance_to_actor)

        if nearest_defender_distance == 999.0:
            nearest_defender_distance = 0.0

    shot = event.get("shot", {})
    body_part_raw = shot.get("body_part", {})
    body_part = body_part_raw.get("name") if isinstance(body_part_raw, dict) else str(body_part_raw or "")

    return {
        "distance": distance,
        "angle": angle,
        "log_distance": math.log(distance + 1),
        "distance_squared": distance**2,
        "angle_distance_interaction": angle * distance,
        "pressure_weighted_distance": distance * (1 + defender_count),
        "defender_count": defender_count,
        "nearest_defender_distance": nearest_defender_distance,
        "goalkeeper_distance": goalkeeper_distance,
        "has_360": int(bool(freeze_frame)),
        "body_part": body_part,
    }


def compute_xg(event: dict) -> float:
    features = _extract_features_from_event(event)
    if features is None:
        return 0.0

    feature_order = [
        "distance",
        "angle",
        "log_distance",
        "distance_squared",
        "angle_distance_interaction",
        "pressure_weighted_distance",
        "defender_count",
        "nearest_defender_distance",
        "goalkeeper_distance",
        "has_360",
        "body_part",
    ]
    X = pd.DataFrame([[features[column] for column in feature_order]], columns=feature_order)

    model = load_xg_model()
    if model is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                xg_value = float(model.predict_proba(X)[0][1])
            return min(max(xg_value, 0.0), 1.0)
        except Exception:
            pass

    xg_value = max(0.0, 0.4 - 0.01 * features["distance"])
    return min(max(xg_value, 0.0), 1.0)
