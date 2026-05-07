"""Decision Quality model loader and predictor.

== HOW TO PLUG IN YOUR TRAINED MODEL ==

1. Train your LightGBM / MLP / LogReg model in Notebook 07 and save:
       joblib.dump(model,  'dq_model.pkl')
       joblib.dump(scaler, 'dq_scaler.pkl')

2. Copy both files into:
       backend/data/dq_model.pkl
       backend/data/dq_scaler.pkl

3. Restart the backend — the model loads automatically on first prediction.

The model must accept a [N × 22] float matrix with features in this order
(same as FEATURE_COLS in decision_quality.py):

  ball_x, ball_y, dist_to_goal, angle_to_goal,
  nearest_defender_dist, num_defenders_close, opponent_density,
  defensive_compactness, nearest_teammate_dist, defenders_ahead,
  type_dribble, type_pass, type_shot,
  target_x, target_y, distance,
  cand_nearest_def_dist, cand_avg_top2_def_dist, cand_num_defenders_near,
  cand_num_defenders_in_lane, cand_min_def_dist_to_lane, cand_defenders_ahead

It must expose a sklearn-compatible predict_proba(X) returning [N × 2]
where column 1 = P(success).

While no model file is present, a rule-based heuristic is used that
produces qualitatively reasonable scores but is less accurate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_scaler = None
_attempted_load = False


def _try_load() -> bool:
    global _model, _scaler, _attempted_load
    if _attempted_load:
        return _model is not None

    _attempted_load = True
    data_dir = Path(__file__).parent.parent.parent / "data"
    model_path = data_dir / "dq_model.pkl"
    scaler_path = data_dir / "dq_scaler.pkl"

    if not (model_path.exists() and scaler_path.exists()):
        logger.info(
            "DQ model files not found at %s — using heuristic fallback. "
            "Drop dq_model.pkl + dq_scaler.pkl there to activate the ML model.",
            data_dir,
        )
        return False

    try:
        import joblib  # type: ignore
        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        logger.info("✅ DQ model loaded from %s", model_path)
        return True
    except Exception as exc:
        logger.warning("⚠️  Failed to load DQ model: %s — using heuristic fallback.", exc)
        return False


def predict_success_probability(X_array: np.ndarray) -> np.ndarray:
    """Return P(success) for each row in X_array (shape [N, 22]).

    Uses the trained ML model when available; falls back to a rule-based
    heuristic otherwise.
    """
    if _try_load() and _model is not None and _scaler is not None:
        try:
            X_scaled = _scaler.transform(X_array)
            proba = _model.predict_proba(X_scaled)
            return proba[:, 1].astype(np.float64)
        except Exception as exc:
            logger.warning("⚠️  DQ model prediction failed: %s — using heuristic.", exc)

    return _heuristic(X_array)


def _heuristic(X: np.ndarray) -> np.ndarray:
    """Rule-based P(success) proxy.

    Feature column indices (22 features, matches FEATURE_COLS in decision_quality.py):
      0  ball_x                     11 type_pass
      1  ball_y                     12 type_shot
      2  dist_to_goal               13 target_x
      3  angle_to_goal              14 target_y
      4  nearest_defender_dist      15 distance
      5  num_defenders_close        16 cand_nearest_def_dist
      6  opponent_density           17 cand_avg_top2_def_dist
      7  defensive_compactness      18 cand_num_defenders_near
      8  nearest_teammate_dist      19 cand_num_defenders_in_lane
      9  defenders_ahead            20 cand_min_def_dist_to_lane
     10  type_dribble               21 cand_defenders_ahead
    """
    cand_nearest_def = X[:, 16]   # farther = safer
    lane_blocked = X[:, 19]        # defenders in passing lane
    is_shot = X[:, 12]
    target_x = X[:, 13]            # more forward = higher threat
    ball_x = X[:, 0]

    score = (
        0.35 * np.clip(cand_nearest_def / 0.2, 0.0, 1.0)
        - 0.15 * np.clip(lane_blocked / 3.0, 0.0, 1.0)
        + 0.10 * np.clip(target_x, 0.0, 1.0)
        - 0.10 * is_shot * np.clip(1.0 - target_x, 0.0, 1.0)
        + 0.10 * np.clip(ball_x, 0.0, 1.0)
        + 0.30
    )
    return np.clip(score, 0.05, 0.95).astype(np.float64)
