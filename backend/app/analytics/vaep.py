"""VAEP computation aligned with the training notebook."""
from __future__ import annotations

import math
import warnings
from typing import Optional

import pandas as pd

from app.services.model_loader import load_vaep_models


def _extract_vaep_features(event: dict) -> Optional[dict]:
    location = event.get("location")
    if not location or len(location) < 2:
        return None

    start_x, start_y = float(location[0]), float(location[1])
    event_type = event.get("type", "")
    if isinstance(event_type, dict):
        event_type = event_type.get("name", "")
    event_type = str(event_type)

    end_x, end_y = start_x, start_y
    xt_start = 0.0
    xt_end = 0.0
    delta_xt = 0.0
    success = 1

    if event_type == "Pass":
        pass_data = event.get("pass", {})
        end_loc = pass_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = float(end_loc[0]), float(end_loc[1])
            from app.analytics.xt import compute_xt_delta, get_xt_value

            xt_start = get_xt_value(start_x, start_y)
            xt_end = get_xt_value(end_x, end_y)
            delta_xt = compute_xt_delta(start_x, start_y, end_x, end_y)
        if "outcome" in pass_data:
            success = 0
    elif event_type == "Carry":
        carry_data = event.get("carry", {})
        end_loc = carry_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = float(end_loc[0]), float(end_loc[1])
            from app.analytics.xt import compute_xt_delta, get_xt_value

            xt_start = get_xt_value(start_x, start_y)
            xt_end = get_xt_value(end_x, end_y)
            delta_xt = compute_xt_delta(start_x, start_y, end_x, end_y)
    elif event_type == "Shot":
        shot_data = event.get("shot", {})
        end_loc = shot_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = float(end_loc[0]), float(end_loc[1])

    action_length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    distance_to_goal = math.sqrt((120.0 - start_x) ** 2 + abs(40.0 - start_y) ** 2)
    end_distance_to_goal = math.sqrt((120.0 - end_x) ** 2 + abs(40.0 - end_y) ** 2)

    return {
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "action_length": action_length,
        "distance_to_goal": distance_to_goal,
        "end_distance_to_goal": end_distance_to_goal,
        "xt_start": xt_start,
        "xt_end": xt_end,
        "delta_xt": delta_xt,
        "success": success,
        "action_type": event_type,
    }


def compute_vaep(event: dict) -> float:
    features = _extract_vaep_features(event)
    if features is None:
        return 0.0

    feature_order = [
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "action_length",
        "distance_to_goal",
        "end_distance_to_goal",
        "xt_start",
        "xt_end",
        "delta_xt",
        "success",
        "action_type",
    ]
    X = pd.DataFrame([[features[column] for column in feature_order]], columns=feature_order)

    scoring_model, conceding_model = load_vaep_models()
    if scoring_model is not None and conceding_model is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                p_scoring = float(scoring_model.predict_proba(X)[0][1])
                p_conceding = float(conceding_model.predict_proba(X)[0][1])
            return p_scoring - p_conceding
        except Exception:
            pass

    positional_bonus = max(0.0, (80.0 - features["distance_to_goal"]) / 80.0) * 0.05
    return features["delta_xt"] * 0.3 + positional_bonus
