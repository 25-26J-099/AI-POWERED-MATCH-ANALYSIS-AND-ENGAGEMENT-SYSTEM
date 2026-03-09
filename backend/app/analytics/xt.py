"""Expected Threat (xT) computation — grid-based model."""

import json
import os
import numpy as np
from typing import List, Tuple, Optional

# Standard pitch dimensions (StatsBomb: 120 x 80)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GRID_ROWS = 8
GRID_COLS = 12

_xt_grid: Optional[np.ndarray] = None


def _load_xt_grid() -> np.ndarray:
    """Load the xT grid — supports .npy (numpy) and .json formats."""
    global _xt_grid
    if _xt_grid is not None:
        return _xt_grid

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

    # Priority 1: numpy file (.npy)
    npy_path = os.path.join(data_dir, "xt_grid.npy")
    if os.path.exists(npy_path):
        _xt_grid = np.load(npy_path)
        return _xt_grid

    # Priority 2: JSON file
    json_path = os.path.join(data_dir, "xt_grid.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            _xt_grid = np.array(json.load(f))
        return _xt_grid

    # Fallback: generate a simple gradient grid (higher threat near goal)
    _xt_grid = np.zeros((GRID_ROWS, GRID_COLS))
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            x_factor = (c / (GRID_COLS - 1)) ** 2
            y_center = abs(r - (GRID_ROWS - 1) / 2) / ((GRID_ROWS - 1) / 2)
            y_factor = 1 - 0.3 * y_center
            _xt_grid[r, c] = x_factor * y_factor * 0.15
    return _xt_grid


def location_to_cell(x: float, y: float) -> Tuple[int, int]:
    """Map a pitch location to a grid cell (row, col)."""
    col = min(int(x / PITCH_LENGTH * GRID_COLS), GRID_COLS - 1)
    row = min(int(y / PITCH_WIDTH * GRID_ROWS), GRID_ROWS - 1)
    return max(0, row), max(0, col)


def get_xt_value(x: float, y: float) -> float:
    """Get the xT value for a given location."""
    grid = _load_xt_grid()
    row, col = location_to_cell(x, y)
    return float(grid[row, col])


def compute_xt_delta(start_x: float, start_y: float, end_x: float, end_y: float) -> float:
    """Compute xT delta for a pass or carry: xT(end) - xT(start)."""
    return get_xt_value(end_x, end_y) - get_xt_value(start_x, start_y)


def compute_event_xt(event: dict) -> float:
    """Compute xT for a single event (pass or carry).

    Args:
        event: Raw event dict with 'location' and either 'pass.end_location' or 'carry.end_location'

    Returns:
        xT delta value, or 0.0 if not applicable.
    """
    location = event.get("location")
    if not location or len(location) < 2:
        return 0.0

    start_x, start_y = location[0], location[1]
    event_type = event.get("type", "")
    if isinstance(event_type, dict):
        event_type = event_type.get("name", "")

    # For passes
    if event_type == "Pass":
        pass_data = event.get("pass", {})
        end_loc = pass_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            # Only count successful passes
            outcome = pass_data.get("outcome", {})
            if isinstance(outcome, dict) and outcome.get("name") == "Incomplete":
                return 0.0
            return compute_xt_delta(start_x, start_y, end_loc[0], end_loc[1])

    # For carries
    elif event_type == "Carry":
        carry_data = event.get("carry", {})
        end_loc = carry_data.get("end_location")
        if end_loc and len(end_loc) >= 2:
            return compute_xt_delta(start_x, start_y, end_loc[0], end_loc[1])

    return 0.0


def get_xt_grid_data() -> dict:
    """Return xT grid for frontend visualization."""
    grid = _load_xt_grid()
    return {
        "grid": grid.tolist(),
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
    }
