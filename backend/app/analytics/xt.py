"""Expected Threat (xT) computation — grid-based model.

Grid matches the training notebook (xt-model-2.ipynb):
  N_X = 16  cells along pitch length (first dimension of grid)
  N_Y = 12  cells along pitch width  (second dimension of grid)
  Indexing: grid[cell_x, cell_y]  where cell_x = f(x), cell_y = f(y)
"""

import json
import os
import numpy as np
from typing import List, Tuple, Optional

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
N_X = 16   # pitch-length cells (first grid dimension)
N_Y = 12   # pitch-width cells  (second grid dimension)

_xt_grid: Optional[np.ndarray] = None


def _load_xt_grid() -> np.ndarray:
    """Load the trained xT grid (xt_grid_2.npy, shape 16×12)."""
    global _xt_grid
    if _xt_grid is not None:
        return _xt_grid

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    # Priority 1: trained model file saved by the notebook
    for fname in ("xt_grid_2.npy", "xt_grid.npy"):
        npy_path = os.path.join(data_dir, fname)
        if os.path.exists(npy_path):
            loaded = np.load(npy_path)
            if loaded.shape == (N_X, N_Y):
                _xt_grid = loaded
                return _xt_grid

    # Priority 2: JSON (only accept if shape matches)
    json_path = os.path.join(data_dir, "xt_grid.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            arr = np.array(json.load(f))
        if arr.shape == (N_X, N_Y):
            _xt_grid = arr
            return _xt_grid

    # Fallback: simple gradient grid (higher threat near opponent goal)
    _xt_grid = np.zeros((N_X, N_Y))
    for cx in range(N_X):
        for cy in range(N_Y):
            x_factor = (cx / (N_X - 1)) ** 2
            y_center = abs(cy - (N_Y - 1) / 2) / ((N_Y - 1) / 2)
            y_factor = 1.0 - 0.3 * y_center
            _xt_grid[cx, cy] = x_factor * y_factor * 0.15
    return _xt_grid


def location_to_cell(x: float, y: float) -> Tuple[int, int]:
    """Map a pitch location to (cell_x, cell_y) matching notebook get_cell(x, y)."""
    cell_x = min(int(x / PITCH_LENGTH * N_X), N_X - 1)
    cell_y = min(int(y / PITCH_WIDTH  * N_Y), N_Y - 1)
    return max(0, cell_x), max(0, cell_y)


def get_xt_value(x: float, y: float) -> float:
    """Get the xT value for a given pitch location."""
    grid = _load_xt_grid()
    cell_x, cell_y = location_to_cell(x, y)
    return float(grid[cell_x, cell_y])


def compute_xt_delta(start_x: float, start_y: float, end_x: float, end_y: float) -> float:
    """xT delta for a pass or carry: xT(end) - xT(start)."""
    return get_xt_value(end_x, end_y) - get_xt_value(start_x, start_y)


def compute_event_xt(event: dict) -> float:
    """Compute xT for a single pass or carry event.

    Only counts successful passes (no outcome key = success in StatsBomb format).
    Returns 0.0 for all other event types.
    """
    location = event.get("location")
    if not location or len(location) < 2:
        return 0.0

    start_x, start_y = location[0], location[1]
    event_type = event.get("type", "")
    if isinstance(event_type, dict):
        event_type = event_type.get("name", "")

    if event_type == "Pass":
        pass_data = event.get("pass", {})
        end_loc = pass_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            outcome = pass_data.get("outcome", {})
            if isinstance(outcome, dict) and outcome.get("name") == "Incomplete":
                return 0.0
            return compute_xt_delta(start_x, start_y, end_loc[0], end_loc[1])

    elif event_type == "Carry":
        carry_data = event.get("carry", {})
        end_loc = carry_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            return compute_xt_delta(start_x, start_y, end_loc[0], end_loc[1])

    return 0.0


def get_xt_grid_data() -> dict:
    """Return xT grid for frontend visualization.

    The grid is returned as-is (shape N_X × N_Y, i.e. 16 × 12).
    Frontend should treat axis 0 as the pitch-length direction.
    """
    grid = _load_xt_grid()
    return {
        "grid": grid.tolist(),
        "n_x": N_X,
        "n_y": N_Y,
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
    }
