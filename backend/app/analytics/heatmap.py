"""Heatmap generation from player event locations."""

import numpy as np
from typing import List


# Pitch dimensions (StatsBomb: 120 x 80)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
HEATMAP_BINS_X = 12
HEATMAP_BINS_Y = 8


def compute_heatmap_from_events(events) -> dict:
    """Generate a 2D histogram heatmap from event locations.

    Args:
        events: List of Event ORM objects or raw dicts.

    Returns:
        Dictionary with grid data ready for frontend rendering.
    """
    locations = []

    for e in events:
        # Handle both ORM objects and raw dicts
        if hasattr(e, "x") and e.x is not None and e.y is not None:
            locations.append((e.x, e.y))
        elif isinstance(e, dict):
            loc = e.get("location")
            if loc and len(loc) >= 2:
                locations.append((loc[0], loc[1]))

    if not locations:
        return {
            "grid": [[0] * HEATMAP_BINS_X for _ in range(HEATMAP_BINS_Y)],
            "bins_x": HEATMAP_BINS_X,
            "bins_y": HEATMAP_BINS_Y,
            "max_value": 0,
        }

    xs = [loc[0] for loc in locations]
    ys = [loc[1] for loc in locations]

    # Create 2D histogram
    heatmap, _, _ = np.histogram2d(
        ys, xs,
        bins=[HEATMAP_BINS_Y, HEATMAP_BINS_X],
        range=[[0, PITCH_WIDTH], [0, PITCH_LENGTH]],
    )

    max_val = float(heatmap.max()) if heatmap.max() > 0 else 1.0

    return {
        "grid": heatmap.tolist(),
        "bins_x": HEATMAP_BINS_X,
        "bins_y": HEATMAP_BINS_Y,
        "max_value": max_val,
        "total_touches": len(locations),
    }


def compute_heatmap_from_raw_events(events: List[dict]) -> dict:
    """Generate heatmap from raw event dicts (for convenience)."""
    return compute_heatmap_from_events(events)
