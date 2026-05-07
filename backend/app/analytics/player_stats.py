"""Player statistics aggregation from event data."""

from typing import Dict, List, Optional
from app.analytics.xt import compute_event_xt
from app.analytics.xg import compute_xg
from app.analytics.vaep import compute_vaep, VAEP_ACTION_TYPES


def _get_event_type(event: dict) -> str:
    t = event.get("type", "")
    if isinstance(t, dict):
        return t.get("name", "")
    return str(t)


def _is_successful_pass(event: dict) -> bool:
    pass_data = event.get("pass", {})
    outcome = pass_data.get("outcome", {})
    if isinstance(outcome, dict):
        return outcome.get("name") not in ["Incomplete", "Out", "Unknown"]
    return True  # No outcome key = successful in StatsBomb format


def _is_progressive_pass(event: dict) -> bool:
    """A pass is progressive if it moves ≥9.15 m (10 yards) toward goal."""
    location = event.get("location", [])
    pass_data = event.get("pass", {})
    end_loc = pass_data.get("end_location", [])

    if len(location) < 2 or len(end_loc) < 2:
        return False

    start_dist = 120.0 - location[0]
    end_dist = 120.0 - end_loc[0]

    return (start_dist - end_dist) >= 9.15 and _is_successful_pass(event)


def _is_progressive_carry(event: dict) -> bool:
    """A carry is progressive if it moves ≥9.15 m toward goal."""
    location = event.get("location", [])
    carry_data = event.get("carry", {})
    end_loc = carry_data.get("end_location", [])

    if len(location) < 2 or len(end_loc) < 2:
        return False

    start_dist = 120.0 - location[0]
    end_dist = 120.0 - end_loc[0]

    return (start_dist - end_dist) >= 9.15


def compute_player_stats(
    player_events: List[dict],
    precomputed_vaep: Optional[Dict[str, float]] = None,
) -> Dict:
    """Aggregate all statistics for a single player from their events.

    Args:
        player_events: Raw event dicts for one player (in match order).
        precomputed_vaep: Optional mapping of event id → vaep_value produced by
            compute_match_vaep_values(). When supplied, VAEP is looked up from
            this dict (true sequential formula). When None, falls back to the
            single-event approximation for valid action types.

    Returns:
        Dictionary of aggregated stats.
    """
    stats = {
        "passes": 0,
        "successful_passes": 0,
        "pass_accuracy": 0.0,
        "progressive_passes": 0,
        "carries": 0,
        "progressive_carries": 0,
        "shots": 0,
        "touches": 0,
        "pressures": 0,
        "recoveries": 0,
        "tackles": 0,
        "interceptions": 0,
        "duels_won": 0,
        "duels_total": 0,
        "xg": 0.0,
        "xt": 0.0,
        "vaep": 0.0,
    }

    for event in player_events:
        event_type = _get_event_type(event)

        # Touch count — any event with a location
        if event.get("location"):
            stats["touches"] += 1

        if event_type == "Pass":
            stats["passes"] += 1
            if _is_successful_pass(event):
                stats["successful_passes"] += 1
            if _is_progressive_pass(event):
                stats["progressive_passes"] += 1
            stats["xt"] += compute_event_xt(event)

        elif event_type == "Carry":
            stats["carries"] += 1
            if _is_progressive_carry(event):
                stats["progressive_carries"] += 1
            stats["xt"] += compute_event_xt(event)

        elif event_type == "Shot":
            stats["shots"] += 1
            stats["xg"] += compute_xg(event)

        elif event_type == "Pressure":
            stats["pressures"] += 1

        elif event_type == "Ball Recovery":
            stats["recoveries"] += 1

        elif event_type == "Tackle":
            stats["tackles"] += 1

        elif event_type == "Interception":
            stats["interceptions"] += 1

        elif event_type == "Duel":
            stats["duels_total"] += 1
            duel_data = event.get("duel", {})
            outcome = duel_data.get("outcome", {})
            if isinstance(outcome, dict) and outcome.get("name") in ["Won", "Success"]:
                stats["duels_won"] += 1

        # VAEP — only for the action types included in training
        if event_type in VAEP_ACTION_TYPES:
            event_id = str(event.get("id") or event.get("event_uuid") or "")
            if precomputed_vaep is not None and event_id in precomputed_vaep:
                stats["vaep"] += precomputed_vaep[event_id]
            elif precomputed_vaep is None:
                # Fallback: single-event approximation
                stats["vaep"] += compute_vaep(event)

    if stats["passes"] > 0:
        stats["pass_accuracy"] = round(stats["successful_passes"] / stats["passes"] * 100, 2)

    return stats
