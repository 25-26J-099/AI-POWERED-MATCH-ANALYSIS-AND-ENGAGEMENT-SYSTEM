"""Service to parse combined tracking JSON into the two formats expected by the commentary scripts."""
import json
from typing import Dict, List, Any

def parse_combined_events(combined_events: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Takes the combined Component 1 JSON list and splits it into:
    1. event_json (list of events without freeze_frame restructuring needed, just as is)
    2. three_sixty_json (list of extracted freeze frames with event_uuid mapped)
    """
    event_json = combined_events
    three_sixty_json = []

    for event in combined_events:
        if "freeze_frame" in event and event["freeze_frame"]:
            ff = event["freeze_frame"]
            # Component 1 can emit freeze_frame as a list directly or as {"players": [...]}
            players = ff.get("players", []) if isinstance(ff, dict) else ff
            
            three_sixty_json.append({
                "event_uuid": event.get("id") or str(event.get("event_uuid", "")),
                "freeze_frame": players
            })

    return event_json, three_sixty_json
