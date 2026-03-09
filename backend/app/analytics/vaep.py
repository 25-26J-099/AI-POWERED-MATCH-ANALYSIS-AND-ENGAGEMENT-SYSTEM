"""VAEP (Valuing Actions by Estimating Probabilities) computation."""

import math
import numpy as np
from typing import Optional
from app.analytics.xt import get_xt_value
from app.services.model_loader import load_vaep_models


def _extract_vaep_features(event: dict, success: bool = True) -> Optional[np.ndarray]:
    """Extract VAEP feature vector from an action event.

    Features (12):
        start_x, start_y, end_x, end_y, action_length,
        distance_to_goal, end_distance_to_goal,
        xt_start, xt_end, delta_xt, success, action_type
    """
    location = event.get("location")
    if not location or len(location) < 2:
        return None

    start_x, start_y = location[0], location[1]

    # Determine end location
    event_type = event.get("type", "")
    if isinstance(event_type, dict):
        event_type = event_type.get("name", "")

    end_x, end_y = start_x, start_y  # default

    if event_type == "Pass":
        end_loc = event.get("pass", {}).get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = end_loc[0], end_loc[1]
        outcome = event.get("pass", {}).get("outcome", {})
        if isinstance(outcome, dict) and outcome.get("name") == "Incomplete":
            success = False
    elif event_type == "Carry":
        end_loc = event.get("carry", {}).get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = end_loc[0], end_loc[1]
    elif event_type == "Shot":
        end_loc = event.get("shot", {}).get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = end_loc[0], end_loc[1]
        outcome = event.get("shot", {}).get("outcome", {})
        if isinstance(outcome, dict) and outcome.get("name") in ["Saved", "Blocked", "Off T", "Wayward"]:
            success = False

    # Goal coordinates
    goal_x, goal_y = 120.0, 40.0

    action_length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    distance_to_goal = math.sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2)
    end_distance_to_goal = math.sqrt((goal_x - end_x) ** 2 + (goal_y - end_y) ** 2)

    xt_start = get_xt_value(start_x, start_y)
    xt_end = get_xt_value(end_x, end_y)
    delta_xt = xt_end - xt_start

    # Action type encoding
    action_type_map = {
        "Pass": 0, "Carry": 1, "Shot": 2, "Dribble": 3,
        "Pressure": 4, "Ball Recovery": 5, "Interception": 6,
        "Clearance": 7, "Foul Committed": 8, "Duel": 9,
    }
    action_code = action_type_map.get(event_type, 10)

    return np.array([
        start_x, start_y, end_x, end_y,
        action_length, distance_to_goal, end_distance_to_goal,
        xt_start, xt_end, delta_xt,
        float(success), float(action_code),
    ])


def compute_vaep(event: dict) -> float:
    """Compute VAEP value for a single action.

    VAEP = P(scoring_next) - P(conceding_next)

    Uses two pre-trained models loaded from HuggingFace.
    Falls back to a simplified xT-based estimate if models unavailable.
    """
    features = _extract_vaep_features(event)
    if features is None:
        return 0.0

    X = features.reshape(1, -1)

    scoring_model, conceding_model = load_vaep_models()
    if scoring_model is not None and conceding_model is not None:
        try:
            p_scoring = float(scoring_model.predict_proba(X)[0][1])
            p_conceding = float(conceding_model.predict_proba(X)[0][1])
            return p_scoring - p_conceding
        except Exception:
            pass

    # Fallback: simplified estimate based on xT delta and position
    delta_xt = features[9]  # delta_xt
    dist_to_goal = features[5]  # distance_to_goal

    # Actions closer to goal and with positive xT delta are more valuable
    positional_bonus = max(0, (80 - dist_to_goal) / 80) * 0.05
    return delta_xt * 0.3 + positional_bonus
