"""VAEP computation aligned with the training notebooks.

Key alignment notes vs VAEPdataset-2.ipynb / value_models_4_vaep_training-2.ipynb:

  Feature alignment:
  - xt_start / xt_end / delta_xt computed for ALL valid action types, not just Pass/Carry
  - Shot success = 1 only when outcome == "Goal" (matches training)
  - Valid actions: Pass, Carry, Shot, Dribble, Interception, Clearance

  Formula alignment:
  - Training VAEP = (P_score_next − P_score) − (P_concede_next − P_concede)
  - compute_match_vaep_values() implements this correctly by processing all match
    events in sequence and computing the delta between consecutive actions.
  - compute_vaep() (single-event) is kept as a per-event fallback only.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import pandas as pd

from app.services.model_loader import load_vaep_models

# Action types included in VAEP training dataset (VAEPdataset-2.ipynb)
VAEP_ACTION_TYPES = {"Pass", "Carry", "Shot", "Dribble", "Interception", "Clearance"}

_FEATURE_ORDER = [
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


def _get_event_type(event: dict) -> str:
    t = event.get("type", "")
    if isinstance(t, dict):
        return t.get("name", "")
    return str(t)


def _extract_vaep_features(event: dict) -> Optional[dict]:
    """Extract VAEP features for a single event, matching VAEPdataset-2.ipynb."""
    from app.analytics.xt import get_xt_value, compute_xt_delta

    location = event.get("location")
    if not location or len(location) < 2:
        return None

    start_x, start_y = float(location[0]), float(location[1])
    event_type = _get_event_type(event)

    if event_type not in VAEP_ACTION_TYPES:
        return None

    end_x, end_y = start_x, start_y
    success = 1

    if event_type == "Pass":
        pass_data = event.get("pass", {})
        end_loc = pass_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = float(end_loc[0]), float(end_loc[1])
        # Any outcome key present = incomplete pass (VAEPdataset-2.ipynb logic)
        if "outcome" in pass_data:
            success = 0

    elif event_type == "Carry":
        carry_data = event.get("carry", {})
        end_loc = carry_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = float(end_loc[0]), float(end_loc[1])

    elif event_type == "Shot":
        shot_data = event.get("shot", {})
        end_loc = shot_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            end_x, end_y = float(end_loc[0]), float(end_loc[1])
        # success = 1 only for Goal (VAEPdataset-2.ipynb: success = 1 if outcome == "Goal" else 0)
        outcome = shot_data.get("outcome", {})
        outcome_name = outcome.get("name", "") if isinstance(outcome, dict) else str(outcome or "")
        success = 1 if outcome_name == "Goal" else 0

    # For Dribble, Interception, Clearance: end = start, success = 1 (training default)

    # xT features — computed for all action types, matching VAEPdataset-2.ipynb
    xt_start = get_xt_value(start_x, start_y)
    xt_end = get_xt_value(end_x, end_y)
    delta_xt = xt_end - xt_start

    action_length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

    # Goal centre at (120, 40) in StatsBomb units (VAEPdataset-2.ipynb: GOAL_X=120, GOAL_Y=40)
    distance_to_goal = math.sqrt((120.0 - start_x) ** 2 + (40.0 - start_y) ** 2)
    end_distance_to_goal = math.sqrt((120.0 - end_x) ** 2 + (40.0 - end_y) ** 2)

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


def compute_match_vaep_values(all_events: list[dict]) -> dict[str, float]:
    """Compute VAEP for every action in a full match using sequential deltas.

    Implements the training formula from value_models_4_vaep_training-2.ipynb:
        vaep_value = (P_score_next − P_score) − (P_concede_next − P_concede)

    Args:
        all_events: All raw event dicts for the match, in chronological order
                    (sorted by period, minute, second before calling).

    Returns:
        Mapping of event "id" → vaep_value for each actionable event.
        Returns {} if models are not loaded (caller should fall back to per-event).
    """
    scoring_model, conceding_model = load_vaep_models()
    if scoring_model is None or conceding_model is None:
        return {}

    # Extract features for every valid action, keeping the event id for attribution
    action_ids: list[str] = []
    feature_rows: list[list] = []

    for event in all_events:
        features = _extract_vaep_features(event)
        if features is None:
            continue
        event_id = str(event.get("id") or event.get("event_uuid") or "")
        action_ids.append(event_id)
        feature_rows.append([features[col] for col in _FEATURE_ORDER])

    if not action_ids:
        return {}

    X = pd.DataFrame(feature_rows, columns=_FEATURE_ORDER)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            p_scores = scoring_model.predict_proba(X)[:, 1]
            p_concedes = conceding_model.predict_proba(X)[:, 1]
    except Exception:
        return {}

    # Compute delta between consecutive actions (shift(-1) in notebook)
    n = len(action_ids)
    vaep_map: dict[str, float] = {}

    for i, event_id in enumerate(action_ids):
        if i < n - 1:
            p_score_next = p_scores[i + 1]
            p_concede_next = p_concedes[i + 1]
        else:
            # Last action: no next action → delta = 0
            p_score_next = p_scores[i]
            p_concede_next = p_concedes[i]

        vaep_val = float((p_score_next - p_scores[i]) - (p_concede_next - p_concedes[i]))
        if event_id:
            vaep_map[event_id] = vaep_val

    return vaep_map


def compute_vaep(event: dict) -> float:
    """Single-event VAEP fallback (used when match-level computation is unavailable).

    Note: this cannot compute the true sequential delta; it returns p_scoring − p_conceding
    as an approximation of the action's value.
    """
    features = _extract_vaep_features(event)
    if features is None:
        return 0.0

    X = pd.DataFrame([[features[col] for col in _FEATURE_ORDER]], columns=_FEATURE_ORDER)

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
