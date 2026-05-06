#!/usr/bin/env python3
"""
Play-by-Play Commentary Generation Pipeline
=============================================
Processes StatsBomb event JSON + 360 data to:
1. Detect commentary-worthy anchor events
2. Generate flattened event streams matching training format
3. Classify intensity (Neutral / High / Goal)
4. Generate commentary via fine-tuned LLM (Ollama)
"""

import json
import math
import random
import sys
import os
import re
import requests
from datetime import datetime
from typing import Any

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "pbp-llama"
EVENT_JSON_PATH = "demo1_event.json"
THREESIXTY_JSON_PATH = "demo1_threesixty.json"
CONTEXT_BEFORE = 5  # events before anchor
CONTEXT_AFTER = 2   # events after anchor
RANDOM_SEED = os.getenv("PBP_RANDOM_SEED")
PASS_SEQUENCE_MIN_PASSES = 4
PASS_SEQUENCE_MIN_SECONDS = 10.0
PASS_SEQUENCE_NAMECALL_GAP_SECONDS = 3.5
EVEN_COVERAGE_BUCKET_SECONDS = 12.0
MIN_ANCHOR_GAP_SECONDS = 3.0
PRE_GOAL_SUPPRESS_SECONDS = 6.0
PASS_NAMECALL_CHANCE = 0.70
PASS_SILENCE_CHANCE = 0.20

if RANDOM_SEED is not None:
    try:
        random.seed(int(RANDOM_SEED))
    except ValueError:
        random.seed(RANDOM_SEED)

# ──────────────────────────────────────────────
# LOCATION HELPERS (StatsBomb 120×80 pitch)
# ──────────────────────────────────────────────

def get_third(x: float | None) -> str:
    if x is None: return "unknown third"
    if x < 40: return "defensive third"
    elif x < 80: return "middle third"
    else: return "final third"

def get_vertical_lane(y: float | None) -> str:
    if y is None: return "unknown lane"
    if y < 18: return "left wing"
    elif y < 30: return "left half-space"
    elif y < 50: return "central lane"
    elif y < 62: return "right half-space"
    else: return "right wing"

def is_in_penalty_box(x: float | None, y: float | None) -> bool:
    if x is None or y is None: return False
    return x >= 102 and 18 <= y <= 62

def is_in_six_yard_box(x: float | None, y: float | None) -> bool:
    if x is None or y is None: return False
    return x >= 114 and 30 <= y <= 50

def distance_to_goal(x: float | None, y: float | None) -> float | None:
    if x is None or y is None: return None
    return math.sqrt((120 - x) ** 2 + (40 - y) ** 2)

def goal_distance_band(d: float | None) -> str:
    if d is None: return "unknown range"
    if d < 6: return "point-blank range"
    elif d < 12: return "very close range"
    elif d < 20: return "close range"
    elif d < 28: return "mid range"
    else: return "long range"

def describe_location(x: float | None, y: float | None) -> str:
    if x is None or y is None:
        return "an unknown location"
    third = get_third(x)
    lane = get_vertical_lane(y)
    tags = [third, lane]
    if is_in_penalty_box(x, y): tags.append("inside the penalty area")
    if is_in_six_yard_box(x, y): tags.append("inside the six-yard box")
    d = distance_to_goal(x, y)
    if d is not None: tags.append(goal_distance_band(d))
    return ", ".join(tags)

def get_location(event: dict) -> tuple[float | None, float | None]:
    loc = event.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None, None

# ──────────────────────────────────────────────
# EVENT HELPERS
# ──────────────────────────────────────────────

def get_event_type(event: dict) -> str:
    return event.get("type", {}).get("name", "Unknown")

def get_player_name(event: dict) -> str:
    player = event.get("player", {})
    if isinstance(player, dict):
        return str(player.get("name") or "Unknown Player")
    return str(player or "Unknown Player")

def get_pass_recipient_name(event: dict) -> str:
    recipient = event.get("pass", {}).get("recipient", {})
    if isinstance(recipient, dict):
        return str(recipient.get("name") or "space")
    return str(recipient or "space")

def get_last_name(full_name: str) -> str:
    parts = [p for p in str(full_name).strip().split() if p]
    if not parts:
        return "Unknown"
    return parts[-1]

def get_player_last_name(event: dict) -> str:
    """Return the exact StatsBomb player label used by the CV export."""
    return get_player_name(event)

def get_pass_recipient_last_name(event: dict) -> str:
    return get_pass_recipient_name(event)

def get_team_name(event: dict) -> str:
    team = event.get("team", {})
    if isinstance(team, dict):
        return str(team.get("name") or "Unknown Team")
    return str(team or "Unknown Team")

def parse_timestamp(ts: str) -> float:
    try:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0

def event_timestamp_seconds(event: dict) -> float:
    return parse_timestamp(event.get("timestamp", "00:00:00.000"))

def is_goal_event(event: dict) -> bool:
    if get_event_type(event) != "Shot":
        return False
    outcome = event.get("shot", {}).get("outcome", {}).get("name", "")
    return outcome == "Goal"

def is_free_kick_event(event: dict) -> bool:
    etype = get_event_type(event)
    if etype == "Pass":
        ptype = event.get("pass", {}).get("type", {}).get("name", "")
        return "Free Kick" in ptype
    if etype == "Shot":
        stype = event.get("shot", {}).get("type", {}).get("name", "")
        return "Free Kick" in stype
    return False

def is_corner_event(event: dict) -> bool:
    play_pattern = event.get("play_pattern", {}).get("name", "")
    if "Corner" in str(play_pattern):
        return True
    ptype = event.get("pass", {}).get("type", {}).get("name", "")
    stype = event.get("shot", {}).get("type", {}).get("name", "")
    return "Corner" in str(ptype) or "Corner" in str(stype)

def is_long_form_commentary_event(event: dict) -> bool:
    etype = get_event_type(event)
    if is_goal_event(event):
        return True
    if etype == "Shot":
        return True
    if etype in ("Foul Committed", "Foul Won"):
        return True
    if is_free_kick_event(event):
        return True
    return False

def time_delta_phrase(delta: float) -> str:
    delta = round(delta, 2)
    if delta <= 0.5:
        return f"Almost instantly ({delta}s later), "
    elif delta <= 2.0:
        return f"Just {delta}s later, "
    elif delta <= 5.0:
        return f"A moment later ({delta}s), "
    else:
        return f"After {delta}s, "

# ──────────────────────────────────────────────
# 360 DATA HELPERS
# ──────────────────────────────────────────────

def build_360_index(threesixty_data: list[dict]) -> dict[str, dict]:
    """Build a lookup from event_uuid -> 360 frame."""
    idx: dict[str, dict] = {}
    for frame in threesixty_data:
        uid = frame.get("event_uuid")
        if uid:
            idx[uid] = frame
    return idx

def count_opponents_near(frame_360: dict | None, actor_x: float | None, actor_y: float | None, radius: float) -> int:
    """Count non-teammate, non-keeper players within radius of actor location from 360 data."""
    if frame_360 is None or actor_x is None or actor_y is None:
        return 0
    count = 0
    for p in frame_360.get("freeze_frame", []):
        if p.get("teammate", False) or p.get("actor", False):
            continue
        loc = p.get("location")
        if isinstance(loc, list) and len(loc) >= 2:
            dist = math.sqrt((actor_x - loc[0]) ** 2 + (actor_y - loc[1]) ** 2)
            if dist <= radius:
                count += 1
    return count

def count_total_opponents(frame_360: dict | None) -> int:
    if frame_360 is None:
        return 0
    return sum(1 for p in frame_360.get("freeze_frame", [])
               if not p.get("teammate", False) and not p.get("actor", False))

def summarize_freeze_frame_from_360(frame_360: dict | None, x: float | None, y: float | None) -> str:
    """Generate defense context string from 360 data."""
    if frame_360 is None:
        return ""
    close = count_opponents_near(frame_360, x, y, 5.0)
    total_opp = count_total_opponents(frame_360)
    return f"Defense context: {close} opponent(s) within 5 units. Total {total_opp} defenders visible."

def summarize_freeze_frame_from_shot(shot_event: dict) -> str:
    """Fallback: use shot's built-in freeze_frame if no 360 data."""
    freeze = shot_event.get("shot", {}).get("freeze_frame", [])
    if not freeze:
        return ""
    sx, sy = get_location(shot_event)
    opponents = 0
    close_opponents = 0
    for p in freeze:
        if p.get("teammate", False):
            continue
        opponents += 1
        loc = p.get("location")
        if isinstance(loc, list) and len(loc) >= 2 and sx is not None and sy is not None:
            dist = math.sqrt((sx - loc[0]) ** 2 + (sy - loc[1]) ** 2)
            if dist <= 5:
                close_opponents += 1
    return f"Defense context: {close_opponents} opponent(s) within 5 units. Total {opponents} defenders visible."

# ──────────────────────────────────────────────
# PASS COMMENTARY TEMPLATES
# ──────────────────────────────────────────────

SHORT_PASS_NAMECALL = [
    "{player}.",
    "{player} to {recipient}.",
    "{player}, {recipient}.",
    "{player} keeps it moving.",
    "{player} finds {recipient}.",
]

LONG_PASS_DESCRIPTIVE = [
    "{player} goes long to {recipient}.",
    "{player} switches it to {recipient}.",
    "{player} clips one into {recipient}.",
    "{player} launches it for {recipient}.",
]

THROUGH_BALL_TEMPLATES = [
    "{player} slips {recipient} in.",
    "{player} threads one through to {recipient}.",
    "{player} unlocks it for {recipient}.",
    "Through ball by {player} for {recipient}.",
]

ASSIST_TEMPLATES = [
    "{player} with the cross.",
    "{player} picks out {recipient}.",
    "{player} hangs it up for {recipient}.",
    "{player} delivers to {recipient}.",
]

SHORT_PASS_LINK_TEMPLATES = [
    "{player} gets it from {from_player}.",
    "{from_player} into {player}.",
    "{player} takes over from {from_player}.",
    "{from_player} finds {player}.",
]

def format_pass_narrative(event: dict, is_namecall: bool = False) -> str:
    """Generate pass text for flattened stream context."""
    p = event.get("pass", {})
    player = get_player_name(event)
    recipient = p.get("recipient", {}).get("name", "space")
    length = round(p.get("length", 0.0), 1)
    height = p.get("height", {}).get("name", "pass").lower()
    outcome = p.get("outcome", {}).get("name", "Complete")
    through_ball = p.get("through_ball", False)
    goal_assist = p.get("goal_assist", False)
    cross = p.get("cross", False)

    # Use special templates for special passes
    if through_ball:
        tmpl = random.choice(THROUGH_BALL_TEMPLATES)
        return tmpl.format(player=player, recipient=recipient)
    
    if goal_assist or cross:
        tmpl = random.choice(ASSIST_TEMPLATES)
        return tmpl.format(player=player, recipient=recipient)
    
    # Name-call for short passes in sequence
    if is_namecall and length < 15:
        tmpl = random.choice(SHORT_PASS_NAMECALL)
        return tmpl.format(player=player, recipient=recipient)

    if outcome in ("Complete", "Unknown", None):
        if length >= 15:
            tmpl = random.choice(LONG_PASS_DESCRIPTIVE)
            return tmpl.format(player=player, recipient=recipient,
                              length=length, height=height)
        else:
            return f"{player} moves it to {recipient}."
    else:
        return f"{player} looks for {recipient}, but it's {outcome}."

def find_previous_team_pass(events: list[dict], event_idx: int, lookback_seconds: float = 8.0) -> dict | None:
    if event_idx <= 0 or event_idx >= len(events):
        return None

    team_id = events[event_idx].get("team", {}).get("id")
    curr_ts = event_timestamp_seconds(events[event_idx])

    for i in range(event_idx - 1, -1, -1):
        evt = events[i]
        if event_timestamp_seconds(evt) + lookback_seconds < curr_ts:
            break
        if get_event_type(evt) != "Pass":
            continue
        if evt.get("team", {}).get("id") == team_id:
            return evt
    return None

def generate_short_pass_commentary(event: dict, prev_team_pass: dict | None) -> str:
    """Short pass line with randomized name-calls and occasional silence."""
    p = event.get("pass", {})
    player = get_player_last_name(event)
    recipient = get_pass_recipient_last_name(event)
    length = float(p.get("length", 0.0) or 0.0)
    through_ball = p.get("through_ball", False)
    goal_assist = p.get("goal_assist", False)
    cross = p.get("cross", False)

    if through_ball:
        return random.choice(THROUGH_BALL_TEMPLATES).format(player=player, recipient=recipient)
    if goal_assist or cross:
        return random.choice(ASSIST_TEMPLATES).format(player=player, recipient=recipient)
    if length >= 20:
        return random.choice(LONG_PASS_DESCRIPTIVE).format(player=player, recipient=recipient, length=round(length, 1))

    roll = random.random()
    if roll < PASS_NAMECALL_CHANCE:
        return random.choice(SHORT_PASS_NAMECALL).format(player=player, recipient=recipient)
    if roll < PASS_NAMECALL_CHANCE + PASS_SILENCE_CHANCE:
        return ""

    if prev_team_pass is not None:
        from_player = get_player_last_name(prev_team_pass)
    else:
        from_player = recipient
    if from_player == player:
        from_player = recipient

    return random.choice(SHORT_PASS_LINK_TEMPLATES).format(
        player=player, recipient=recipient, from_player=from_player
    )

# ──────────────────────────────────────────────
# ANCHOR EVENT DETECTION
# ──────────────────────────────────────────────

# Event types that are always anchored
NO_BRAINER_TYPES = {"Shot", "Foul Committed", "Foul Won", "Half Start", "Half End"}

# Goalkeeper sub-types worth anchoring
GK_ANCHOR_SUBTYPES = {"Save", "Goal Conceded", "Punch", "Keeper Pick-Up"}

def is_no_brainer_anchor(event: dict) -> tuple[bool, str]:
    """Check if event is always commentary-worthy."""
    etype = get_event_type(event)
    
    if etype in NO_BRAINER_TYPES:
        return True, "No-Brainer Core Event"
    
    if etype == "Goal Keeper":
        gk = event.get("goalkeeper", {})
        gk_type = gk.get("type", {}).get("name", "")
        gk_outcome = gk.get("outcome", {}).get("name", "")
        if gk_type in GK_ANCHOR_SUBTYPES or gk_outcome in {"No Touch"}:
            return True, "Goalkeeper Action"
        return False, ""
    
    return False, ""

def is_conditional_anchor(event: dict) -> tuple[bool, str]:
    """Check for high-priority conditional anchors."""
    etype = get_event_type(event)
    x, y = get_location(event)
    
    if etype == "Pass":
        p = event.get("pass", {})
        if p.get("goal_assist"):
            return True, "Goal Assist"
        if p.get("cross"):
            return True, "Cross into Box"
        end_loc = p.get("end_location", [])
        if len(end_loc) >= 2 and end_loc[0] >= 102:
            return True, "Pass into Penalty Area"
        if p.get("through_ball"):
            return True, "Through Ball"
    
    if etype == "Dribble" and x is not None and x >= 80:
        return True, "Dribble in Final Third"
    
    if etype in ("Interception", "Ball Recovery") and x is not None and x >= 80:
        return True, "Defensive Action in Final Third"
    
    if etype == "Clearance" and x is not None and x < 40:
        return True, "Clearance in Defensive Third"
    
    if etype == "Miscontrol" and x is not None and x >= 102:
        return True, "Miscontrol in Penalty Area"
    
    return False, ""

def is_probabilistic_anchor(event: dict) -> tuple[bool, str]:
    """Randomly select some tactical events as anchors for variety."""
    etype = get_event_type(event)
    x, y = get_location(event)
    
    if etype in ("Interception", "Ball Recovery"):
        if random.random() < 0.30:
            return True, "Random Tactical Anchor"
    
    if etype == "Clearance" and x is not None and x >= 40:
        if random.random() < 0.20:
            return True, "Random Clearance Anchor"
    
    if etype == "Dribble" and x is not None and x < 80:
        if random.random() < 0.15:
            return True, "Random Dribble Anchor"
    
    if etype == "Block":
        if random.random() < 0.25:
            return True, "Random Block Anchor"
    
    return False, ""

# ──────────────────────────────────────────────
# INTENSITY CLASSIFICATION
# ──────────────────────────────────────────────

def classify_intensity(event: dict, frame_360: dict | None) -> str:
    """Classify event intensity: Goal, High, or Neutral."""
    etype = get_event_type(event)
    x, y = get_location(event)
    
    # Goal intensity
    if etype == "Shot":
        outcome = event.get("shot", {}).get("outcome", {}).get("name", "")
        if outcome == "Goal":
            return "Goal"
    
    # High intensity conditions
    if etype == "Shot":
        return "High"
    
    if x is not None and x >= 102:
        return "High"
    
    if etype in ("Foul Committed", "Foul Won") and x is not None and x >= 80:
        return "High"
    
    if etype == "Dribble" and x is not None and x >= 80:
        if event.get("under_pressure"):
            return "High"
    
    # 360-enhanced: high pressure near goal
    if x is not None and x >= 90 and frame_360 is not None:
        close_opp = count_opponents_near(frame_360, x, y, 10.0)
        if close_opp >= 3:
            return "High"
    
    if etype == "Goal Keeper":
        gk_type = event.get("goalkeeper", {}).get("type", {}).get("name", "")
        if gk_type in ("Save", "Goal Conceded"):
            return "High"
    
    if etype == "Clearance" and x is not None and x < 30:
        return "High"
    
    if etype == "Block" and x is not None and (x < 25 or x >= 102):
        return "High"
    
    return "Neutral"

# ──────────────────────────────────────────────
# FLATTENED EVENT NARRATIVE GENERATION
# ──────────────────────────────────────────────

def generate_event_narrative(
    event: dict,
    prev_event: dict | None,
    is_anchor: bool,
    frame_360: dict | None,
    is_pass_namecall: bool = False,
) -> str:
    """Generate a single event narrative line matching training format."""
    etype = get_event_type(event)
    player = get_player_name(event)
    team = get_team_name(event)
    x, y = get_location(event)
    zone = describe_location(x, y)
    
    # Time delta
    time_context = ""
    if prev_event:
        prev_ts = parse_timestamp(prev_event.get("timestamp", "0:0:0"))
        curr_ts = parse_timestamp(event.get("timestamp", "0:0:0"))
        delta = round(curr_ts - prev_ts, 2)
        if delta < 0:
            delta = 0.0
        time_context = time_delta_phrase(delta)
    
    pressure = " under heavy pressure" if event.get("under_pressure") else ""
    anchor_flag = "[ANCHOR] " if is_anchor else ""
    
    narrative = f"{anchor_flag}{time_context}{player} ({team}){pressure}"
    
    if etype == "Pass":
        if is_pass_namecall or (not is_anchor and event.get("pass", {}).get("length", 999) < 15):
            pass_text = format_pass_narrative(event, is_namecall=True)
            return f"{anchor_flag}{time_context}{pass_text}"
        else:
            p = event.get("pass", {})
            recipient = p.get("recipient", {}).get("name", "space")
            height = p.get("height", {}).get("name", "pass").lower()
            length = round(p.get("length", 0.0), 1)
            outcome = p.get("outcome", {}).get("name", "Complete")
            
            if outcome in ("Complete", "Unknown", None):
                narrative += (f" plays a successful {length}-yard {height} "
                            f"from {zone} to {recipient}.")
            else:
                narrative += (f" attempts a {length}-yard {height} "
                            f"from {zone} towards {recipient}, but it is {outcome}.")
    
    elif etype == "Shot":
        s = event.get("shot", {})
        outcome = s.get("outcome", {}).get("name", "Unknown").upper()
        technique = s.get("technique", {}).get("name", "shot").lower()
        body_part = s.get("body_part", {}).get("name", "foot").lower()
        xG = round(s.get("statsbomb_xg", 0.0), 3)
        
        # Try 360 data first, then shot freeze_frame
        freeze_ctx = summarize_freeze_frame_from_360(frame_360, x, y)
        if not freeze_ctx:
            freeze_ctx = summarize_freeze_frame_from_shot(event)
        
        narrative += (f" unleashes a {technique} with their {body_part} "
                     f"from {zone}. {freeze_ctx} The xG is {xG}. Outcome: {outcome}.")
    
    elif etype == "Carry":
        c = event.get("carry", {})
        end_loc = c.get("end_location", [])
        if len(end_loc) >= 2:
            end_zone = describe_location(end_loc[0], end_loc[1])
            narrative += f" carries the ball from {zone} into {end_zone}."
        else:
            narrative += f" carries the ball through {zone}."
    
    elif etype in ("Ball Receipt*", "Ball Recovery"):
        narrative += f" gains control of the ball in {zone}."
    
    elif etype == "Foul Committed":
        card = event.get("foul_committed", {}).get("card", {}).get("name", "no card")
        narrative += f" commits a foul in {zone}, resulting in {card}."
    
    elif etype == "Dribble":
        narrative += f" dribbles through {zone}."
    
    elif etype == "Pressure":
        narrative += f" applies pressure in {zone}."
    
    elif etype == "Goal Keeper":
        narrative += f" handles the goalkeeper action in {zone}."
    
    else:
        narrative += f" is involved in a {etype} in {zone}."
    
    return narrative

# ──────────────────────────────────────────────
# PASS SEQUENCE DETECTION
# ──────────────────────────────────────────────

def detect_pass_sequences(events: list[dict]) -> list[dict]:
    """
    Detect sustained pass sequences and emit anchors across the sequence.
    """
    anchors = []
    i = 0
    
    while i < len(events):
        event = events[i]
        etype = get_event_type(event)
        
        if etype != "Pass":
            i += 1
            continue
        
        possession = event.get("possession")
        team_id = event.get("team", {}).get("id")
        seq_indices = [i]
        
        j = i + 1
        while j < len(events):
            next_event = events[j]
            next_type = get_event_type(next_event)
            next_possession = next_event.get("possession")
            next_team = next_event.get("team", {}).get("id")
            
            if next_possession != possession:
                break
            
            if next_type in ("Ball Receipt*", "Carry"):
                j += 1
                continue
            
            if next_type == "Pass" and next_team == team_id:
                seq_indices.append(j)
                j += 1
                continue
            
            break
        
        if len(seq_indices) >= PASS_SEQUENCE_MIN_PASSES:
            start_ts = event_timestamp_seconds(events[seq_indices[0]])
            end_ts = event_timestamp_seconds(events[seq_indices[-1]])
            duration = end_ts - start_ts

            if duration >= PASS_SEQUENCE_MIN_SECONDS:
                selected_seq_passes: list[int] = []
                last_pick_ts = -999999.0
                for pass_idx in seq_indices:
                    ts = event_timestamp_seconds(events[pass_idx])
                    if (not selected_seq_passes) or (ts - last_pick_ts >= PASS_SEQUENCE_NAMECALL_GAP_SECONDS):
                        selected_seq_passes.append(pass_idx)
                        last_pick_ts = ts

                if seq_indices[-1] not in selected_seq_passes:
                    selected_seq_passes.append(seq_indices[-1])

                for pass_idx in selected_seq_passes:
                    anchors.append({
                        "index": pass_idx,
                        "event": events[pass_idx],
                        "reason": "Long Pass Sequence Tracker",
                        "pass_sequence_indices": seq_indices,
                        "sequence_duration_seconds": round(duration, 2),
                    })
        
        i = j if j > i + 1 else i + 1
    
    return anchors

def anchor_reason_priority(reason: str) -> int:
    priority = {
        "No-Brainer Core Event": 100,
        "Goal Assist": 95,
        "Goalkeeper Action": 90,
        "Cross into Box": 85,
        "Pass into Penalty Area": 82,
        "Through Ball": 80,
        "Long Pass Sequence Tracker": 70,
        "Even Coverage Anchor": 60,
        "Dribble in Final Third": 55,
        "Defensive Action in Final Third": 50,
        "Clearance in Defensive Third": 45,
        "Miscontrol in Penalty Area": 45,
        "Random Tactical Anchor": 40,
        "Random Clearance Anchor": 35,
        "Random Dribble Anchor": 35,
        "Random Block Anchor": 35,
    }
    return priority.get(reason, 30)

def anchor_priority(anchor: dict) -> int:
    event = anchor["event"]
    base = anchor_reason_priority(anchor.get("reason", ""))
    if is_goal_event(event):
        return base + 200
    if is_long_form_commentary_event(event):
        return base + 100
    return base

def suppress_pre_goal_anchors(anchors: list[dict]) -> list[dict]:
    goal_times = [
        event_timestamp_seconds(a["event"])
        for a in anchors
        if is_goal_event(a["event"])
    ]
    if not goal_times:
        return anchors

    filtered = []
    for a in anchors:
        evt = a["event"]
        if is_goal_event(evt):
            filtered.append(a)
            continue

        ts = event_timestamp_seconds(evt)
        near_goal = any(0 < (goal_ts - ts) <= PRE_GOAL_SUPPRESS_SECONDS for goal_ts in goal_times)
        if near_goal and not is_long_form_commentary_event(evt):
            continue
        filtered.append(a)
    return filtered

def apply_anchor_spacing(anchors: list[dict]) -> list[dict]:
    if len(anchors) <= 1:
        return anchors

    anchors = sorted(anchors, key=lambda a: (event_timestamp_seconds(a["event"]), a["index"]))
    filtered = [anchors[0]]

    for curr in anchors[1:]:
        prev = filtered[-1]
        prev_ts = event_timestamp_seconds(prev["event"])
        curr_ts = event_timestamp_seconds(curr["event"])
        gap = curr_ts - prev_ts

        if gap >= MIN_ANCHOR_GAP_SECONDS:
            filtered.append(curr)
            continue

        prev_long = is_long_form_commentary_event(prev["event"])
        curr_long = is_long_form_commentary_event(curr["event"])
        if prev_long and curr_long:
            filtered.append(curr)
            continue

        if anchor_priority(curr) > anchor_priority(prev):
            filtered[-1] = curr

    filtered.sort(key=lambda a: a["index"])
    return filtered

def pick_even_coverage_candidate(
    events: list[dict],
    bucket_start: float,
    bucket_end: float,
    taken_indices: set[int],
) -> int | None:
    pass_candidates: list[int] = []
    fallback_candidates: list[int] = []

    for i, event in enumerate(events):
        if i in taken_indices:
            continue
        ts = event_timestamp_seconds(event)
        if ts < bucket_start or ts >= bucket_end:
            continue
        etype = get_event_type(event)
        if etype == "Pass":
            pass_candidates.append(i)
        elif etype not in ("Ball Receipt*", "Carry", "Pressure"):
            fallback_candidates.append(i)

    if pass_candidates:
        return pass_candidates[len(pass_candidates) // 2]
    if fallback_candidates:
        return fallback_candidates[0]
    return None

def add_even_coverage_anchors(events: list[dict], anchors: list[dict]) -> list[dict]:
    if not events:
        return anchors

    start_ts = event_timestamp_seconds(events[0])
    end_ts = event_timestamp_seconds(events[-1])
    if end_ts <= start_ts:
        return anchors

    taken_indices = {a["index"] for a in anchors}
    bucket_start = start_ts

    while bucket_start <= end_ts:
        bucket_end = bucket_start + EVEN_COVERAGE_BUCKET_SECONDS
        has_anchor = any(
            bucket_start <= event_timestamp_seconds(a["event"]) < bucket_end
            for a in anchors
        )
        if not has_anchor:
            candidate_idx = pick_even_coverage_candidate(events, bucket_start, bucket_end, taken_indices)
            if candidate_idx is not None:
                anchors.append({
                    "index": candidate_idx,
                    "event": events[candidate_idx],
                    "reason": "Even Coverage Anchor",
                })
                taken_indices.add(candidate_idx)
        bucket_start = bucket_end

    return anchors

# ──────────────────────────────────────────────
# MASTER ANCHOR DETECTION
# ──────────────────────────────────────────────

def detect_all_anchors(events: list[dict], idx_360: dict[str, dict]) -> list[dict]:
    """Detect all anchor events from the full event stream."""
    anchors: list[dict] = []
    anchored_indices: set[int] = set()
    
    # 1) No-brainer + conditional + probabilistic passes
    for i, event in enumerate(events):
        is_nb, reason_nb = is_no_brainer_anchor(event)
        if is_nb:
            anchors.append({"index": i, "event": event, "reason": reason_nb})
            anchored_indices.add(i)
            continue
        
        is_cond, reason_cond = is_conditional_anchor(event)
        if is_cond:
            anchors.append({"index": i, "event": event, "reason": reason_cond})
            anchored_indices.add(i)
            continue
        
        is_prob, reason_prob = is_probabilistic_anchor(event)
        if is_prob:
            anchors.append({"index": i, "event": event, "reason": reason_prob})
            anchored_indices.add(i)
            continue
    
    # 2) Pass sequence anchors (only if not already anchored)
    pass_seq_anchors = detect_pass_sequences(events)
    for psa in pass_seq_anchors:
        idx = psa["index"]
        if idx not in anchored_indices:
            anchors.append(psa)
            anchored_indices.add(idx)
    
    # 3) Add timeline coverage anchors to avoid late clumping
    anchors = add_even_coverage_anchors(events, anchors)

    # 4) Sort by event index
    anchors.sort(key=lambda a: a["index"])
    
    # 5) Deduplicate neighboring anchors
    if len(anchors) > 1:
        filtered = [anchors[0]]
        for a in anchors[1:]:
            prev = filtered[-1]
            if a["index"] - prev["index"] <= 1:
                if anchor_priority(a) > anchor_priority(prev):
                    filtered[-1] = a
            else:
                filtered.append(a)
        anchors = filtered

    # 6) Remove non-key chatter in final seconds before a goal
    anchors = suppress_pre_goal_anchors(anchors)

    # 7) Apply minimum spacing between routine commentary events
    anchors = apply_anchor_spacing(anchors)

    return sorted(anchors, key=lambda a: a["index"])

# ──────────────────────────────────────────────
# CONTEXT WINDOW & STREAM BUILDER
# ──────────────────────────────────────────────

def build_context_window(
    events: list[dict],
    anchor_idx: int,
    anchor_info: dict,
) -> list[dict]:
    """Get events surrounding an anchor for context."""
    possession = events[anchor_idx].get("possession")
    
    # Collect events before (within same possession, up to CONTEXT_BEFORE)
    before = []
    for i in range(max(0, anchor_idx - 20), anchor_idx):
        if events[i].get("possession") == possession:
            before.append(i)
    before = before[-CONTEXT_BEFORE:]  # last N
    
    # Collect events after (within same possession, up to CONTEXT_AFTER)
    after = []
    for i in range(anchor_idx + 1, min(len(events), anchor_idx + 15)):
        if events[i].get("possession") == possession:
            after.append(i)
        if len(after) >= CONTEXT_AFTER:
            break
    
    all_indices = before + [anchor_idx] + after
    return [events[i] for i in all_indices]

def build_flattened_stream(
    context_events: list[dict],
    anchor_event: dict,
    anchor_info: dict,
    idx_360: dict[str, dict],
) -> str:
    """Build the flattened event stream text matching training format."""
    anchor_id = anchor_event.get("id")
    is_pass_seq = anchor_info.get("reason") == "Long Pass Sequence Tracker"
    
    lines = []
    
    for i, event in enumerate(context_events):
        prev_event = context_events[i - 1] if i > 0 else None
        is_anchor = (event.get("id") == anchor_id)
        frame_360 = idx_360.get(event.get("id"))
        
        # Determine if this is a name-call pass
        etype = get_event_type(event)
        is_namecall = False
        if is_pass_seq and etype == "Pass" and not is_anchor:
            pass_len = event.get("pass", {}).get("length", 999)
            if pass_len < 15:
                is_namecall = True
        
        narrative_line = generate_event_narrative(
            event, prev_event, is_anchor, frame_360, is_pass_namecall=is_namecall
        )
        lines.append(narrative_line)
    
    return "\n".join(lines)

# ──────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ──────────────────────────────────────────────

def build_prompt(flattened_stream: str, intensity: str, anchor_event: dict) -> str:
    """Build the LLM prompt, keeping non-goal events concise."""
    parts = [f"Intensity: {intensity}"]
    allowed_names = sorted(collect_allowed_names([anchor_event]))
    if allowed_names:
        parts.append(
            "[NAME RULE: Use only names that appear in the event stream. "
            f"Allowed exact names for this anchor: {', '.join(allowed_names)}. "
            "Do not invent real player names. If the data says Player 17, say Player 17.]"
        )
    parts.append(
        "[FACT RULE: Describe only the action and outcome present in the event stream. "
        "Do not invent corners, crosses, goals, assists, empty nets, saves, or fouls unless they are explicitly shown.]"
    )
    
    if is_goal_event(anchor_event):
        parts.append(
            "[SYSTEM INSTRUCTION: Make the commentary urgent, shorter, "
            "and punchier as this is close to the penalty area or a "
            "high threat situation.]"
        )
        parts.append("")
    elif is_long_form_commentary_event(anchor_event):
        parts.append(
            "[SYSTEM INSTRUCTION: Keep it to one concise line. Focus on the anchor event "
            "only and avoid retelling every preceding pass.]"
        )
        parts.append("")
    else:
        parts.append(
            "[SYSTEM INSTRUCTION: Output an ultra-short call (max 6 words).]"
        )
        parts.append("")
    
    parts.append(f"Flattened Event Stream:")
    parts.append(flattened_stream)
    
    return "\n".join(parts)

def is_llm_commentary_event(event: dict) -> bool:
    etype = get_event_type(event)
    if is_goal_event(event):
        return True
    if etype == "Shot":
        return True
    if etype in ("Foul Committed", "Foul Won"):
        return True
    if is_free_kick_event(event):
        return True
    return False

def generate_template_commentary(events: list[dict], anchor_idx: int, anchor_info: dict) -> str:
    event = events[anchor_idx]
    etype = get_event_type(event)
    player = get_player_last_name(event)

    if is_goal_event(event):
        return random.choice([
            f"Goal by {player}!",
            f"{player} scores!",
            f"{player} finds the net!",
        ])

    if etype == "Shot":
        return random.choice([
            f"Shot by {player}.",
            f"{player} lets one fly.",
            f"{player} gets a shot away.",
        ])

    if is_free_kick_event(event):
        return random.choice([
            f"{player} over the free kick.",
            f"Free kick from {player}.",
            f"{player} takes the set piece.",
        ])

    if etype == "Foul Committed":
        return random.choice([
            f"Foul by {player}.",
            f"{player} brings the man down.",
            f"{player} late there, foul given.",
        ])

    if etype == "Foul Won":
        return random.choice([
            f"{player} wins a foul.",
            f"Foul won by {player}.",
            f"{player} draws the contact.",
        ])

    if etype == "Pass":
        prev_team_pass = find_previous_team_pass(events, anchor_idx)
        return generate_short_pass_commentary(event, prev_team_pass)

    if etype == "Carry":
        return random.choice([
            f"{player} drives forward.",
            f"{player} carries on.",
            f"{player} advances with it.",
        ])

    if etype == "Dribble":
        return random.choice([
            f"{player} takes a touch past one.",
            f"{player} glides forward.",
            f"{player} beats the press.",
        ])

    if etype in ("Interception", "Ball Recovery"):
        return random.choice([
            f"{player} wins it back.",
            f"{player} recovers possession.",
            f"{player} steps in and takes it.",
        ])

    if etype == "Clearance":
        return random.choice([
            f"{player} clears the danger.",
            f"{player} hooks it away.",
            f"{player} gets it out.",
        ])

    if etype == "Goal Keeper":
        return random.choice([
            f"{player} claims it.",
            f"{player} deals with it.",
            f"{player} gathers safely.",
        ])

    if etype == "Pressure":
        return random.choice([
            f"{player} closes quickly.",
            f"{player} presses hard.",
            f"{player} puts pressure on.",
        ])

    reason = anchor_info.get("reason", "")
    if reason == "Even Coverage Anchor":
        return random.choice([
            f"{player} keeps play moving.",
            f"{player} recycles possession.",
        ])

    return f"{player} involved."

_NAME_PHRASE_RE = re.compile(r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)+\b")
_HYphenated_NAME_RE = re.compile(r"\b[A-Z][a-z]+-[A-Z][a-z]+\b")
_SINGLE_NAME_AFTER_ACTION_RE = re.compile(
    r"\b(?:by|from|to|for|finds|find|towards|toward)\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b"
)
_PLAYER_NUMBER_ACTIONS = (
    "advances",
    "beats",
    "brings",
    "carries",
    "claims",
    "clears",
    "closes",
    "deals",
    "draws",
    "drives",
    "finds",
    "gathers",
    "gets",
    "glides",
    "hooks",
    "involved",
    "keeps",
    "late",
    "lets",
    "passes",
    "plays",
    "presses",
    "puts",
    "recycles",
    "scores",
    "takes",
    "wins",
)
_COMMON_CAPITALIZED_PHRASES = {
    "A fine",
    "After",
    "Almost",
    "Ball Recovery",
    "Clearance",
    "Defense",
    "Free Kick",
    "Foul Committed",
    "Foul Won",
    "Goal Keeper",
    "High",
    "Intensity",
    "Just",
    "Player",
    "Regular Play",
    "Shot",
    "The",
    "Unknown Player",
    "Unknown Team",
}


def collect_allowed_names(events: list[dict]) -> set[str]:
    names: set[str] = set()
    for event in events:
        player_name = get_player_name(event)
        if player_name and player_name != "Unknown Player":
            names.add(player_name)
        recipient_name = get_pass_recipient_name(event)
        if recipient_name and recipient_name != "space":
            names.add(recipient_name)
        team_name = get_team_name(event)
        if team_name and team_name != "Unknown Team":
            names.add(team_name)
    return names


def commentary_uses_only_allowed_names(text: str, events: list[dict]) -> bool:
    """Reject LLM output that invents human names absent from the StatsBomb events."""
    if not text:
        return True

    allowed = collect_allowed_names(events)
    allowed_parts = set()
    for name in allowed:
        allowed_parts.add(name)
        allowed_parts.update(part for part in re.split(r"\s+", name) if part)

    for match in _NAME_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip()
        if phrase in allowed or phrase in _COMMON_CAPITALIZED_PHRASES:
            continue
        if phrase.startswith("Player "):
            continue
        return False

    for match in _HYphenated_NAME_RE.finditer(text):
        phrase = match.group(0).strip()
        if phrase not in allowed and phrase not in allowed_parts:
            return False

    generic_player_names = {
        name for name in allowed if re.fullmatch(r"Player\s+\d+", name)
    }
    if generic_player_names and len(generic_player_names) == len([name for name in allowed if name.startswith("Player ")]):
        for match in _SINGLE_NAME_AFTER_ACTION_RE.finditer(text):
            token = match.group(1).strip()
            if token not in allowed_parts and token not in _COMMON_CAPITALIZED_PHRASES:
                return False

    return True


def restore_generic_player_labels(text: str, events: list[dict]) -> str:
    """Turn bare generated labels like '17 shoots' back into exact 'Player 17'."""
    if not text:
        return text
    player_numbers = sorted(
        {
            re.fullmatch(r"Player\s+(\d+)", name).group(1)
            for name in collect_allowed_names(events)
            if re.fullmatch(r"Player\s+\d+", name)
        },
        key=len,
        reverse=True,
    )
    for number in player_numbers:
        text = re.sub(
            rf"(?<!Player\s)\b{re.escape(number)}\b(?=\s+(?:{'|'.join(_PLAYER_NUMBER_ACTIONS)})\b)",
            f"Player {number}",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            rf"\b(by|from|to|for)\s+{re.escape(number)}\b",
            rf"\1 Player {number}",
            text,
            flags=re.IGNORECASE,
        )
    return text


def commentary_respects_anchor_event(text: str, anchor_event: dict) -> bool:
    """Reject obvious event/outcome hallucinations from the LLM."""
    lowered = (text or "").lower()
    etype = get_event_type(anchor_event)

    if not is_goal_event(anchor_event) and re.search(r"\b(goal|scores|empty net|finds the net)\b", lowered):
        return False
    if "executes a" in lowered:
        return False
    if "corner" in lowered and not is_corner_event(anchor_event):
        return False
    if "free kick" in lowered and not is_free_kick_event(anchor_event):
        return False
    if "cross" in lowered and not bool(anchor_event.get("pass", {}).get("cross")):
        return False
    if etype != "Shot" and re.search(r"\bshot|effort|lets one fly\b", lowered):
        return False
    return True


def ensure_allowed_commentary_names(
    text: str,
    events: list[dict],
    anchor_idx: int,
    anchor_info: dict,
) -> str:
    text = restore_generic_player_labels(text, events)
    anchor_event = events[anchor_idx]
    if commentary_uses_only_allowed_names(text, events) and commentary_respects_anchor_event(text, anchor_event):
        return enforce_surname_only(text, events).strip()
    return generate_template_commentary(events, anchor_idx, anchor_info).strip()


def enforce_surname_only(text: str, events: list[dict]) -> str:
    if not text:
        return text

    full_to_last: dict[str, str] = {}
    for event in events:
        player_name = event.get("player", {}).get("name")
        if player_name and not re.fullmatch(r"Player\s+\d+", str(player_name)):
            full_to_last[player_name] = str(player_name)
        recipient_name = event.get("pass", {}).get("recipient", {}).get("name")
        if recipient_name and not re.fullmatch(r"Player\s+\d+", str(recipient_name)):
            full_to_last[recipient_name] = str(recipient_name)

    for full_name in sorted(full_to_last.keys(), key=len, reverse=True):
        last_name = full_to_last[full_name]
        if full_name == last_name:
            continue
        pattern = r"\b" + re.escape(full_name) + r"\b"
        text = re.sub(pattern, last_name, text)

    return text

# ──────────────────────────────────────────────
# OLLAMA INTEGRATION
# ──────────────────────────────────────────────

def check_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def check_model_exists(model_name: str) -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return any(m.get("name", "").startswith(model_name) for m in models)
    except Exception:
        pass
    return False

def create_model(model_name: str, modelfile_path: str) -> bool:
    """Create Ollama model from Modelfile. Tries CLI first, then API."""
    import subprocess
    print(f"Creating model '{model_name}' from {modelfile_path}...")
    modelfile_dir = os.path.dirname(os.path.abspath(modelfile_path))
    
    # Try CLI approach first (handles relative paths naturally)
    ollama_paths = [
        "ollama",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        r"C:\Program Files\Ollama\ollama.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Ollama\ollama.exe"),
    ]
    
    for ollama_bin in ollama_paths:
        try:
            result = subprocess.run(
                [ollama_bin, "create", model_name, "-f", os.path.basename(modelfile_path)],
                cwd=modelfile_dir,
                capture_output=True,
                text=True,
                errors='replace',
                timeout=300,
            )
            print(f"  CLI stdout: {result.stdout.strip()}")
            if result.stderr:
                print(f"  CLI stderr: {result.stderr.strip()}")
            if result.returncode == 0:
                print(f"✅ Model '{model_name}' created successfully via CLI.")
                return True
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"  CLI attempt with '{ollama_bin}' failed: {e}")
            continue
    
    # Fallback: API approach with absolute path resolution
    print("  CLI not found, trying API approach...")
    try:
        with open(modelfile_path, "r") as f:
            modelfile_content = f.read()
        
        # Replace relative FROM path with absolute path
        def resolve_from(match: re.Match) -> str:
            rel_path = match.group(1).strip()
            if not os.path.isabs(rel_path):
                abs_path = os.path.join(modelfile_dir, rel_path)
                return f"FROM {abs_path}"
            return match.group(0)
        
        modelfile_content = re.sub(
            r"^FROM\s+(.+)$", resolve_from, modelfile_content, flags=re.MULTILINE
        )
        
        print(f"  Resolved FROM line: {modelfile_content.split(chr(10))[1] if len(modelfile_content.split(chr(10))) > 1 else 'N/A'}")
        
        r = requests.post(
            f"{OLLAMA_URL}/api/create",
            json={"name": model_name, "modelfile": modelfile_content},
            stream=True,
            timeout=300,
        )
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                status = data.get("status", "")
                if status:
                    print(f"  {status}")
                if "error" in data:
                    print(f"  ERROR: {data['error']}")
                    return False
        print(f"✅ Model '{model_name}' created successfully via API.")
        return True
    except Exception as e:
        print(f"❌ Failed to create model: {e}")
        return False

def generate_commentary(prompt: str) -> str:
    """Call Ollama to generate commentary."""
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 1.5,
                    "min_p": 0.1,
                    "num_predict": 200,
                },
            },
            timeout=120,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        else:
            return f"[ERROR: HTTP {r.status_code}]"
    except Exception as e:
        return f"[ERROR: {e}]"

# ──────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("⚽ Play-by-Play Commentary Generation Pipeline")
    print("=" * 60)
    
    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    event_path = os.path.join(script_dir, EVENT_JSON_PATH)
    threesixty_path = os.path.join(script_dir, THREESIXTY_JSON_PATH)
    modelfile_path = os.path.join(script_dir, "PbpModel", "Modelfile")
    
    # 1) Load data
    print("\n📦 Loading event data...")
    with open(event_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"   Loaded {len(events)} events.")
    
    print("📦 Loading 360 data...")
    with open(threesixty_path, "r", encoding="utf-8") as f:
        threesixty_data = json.load(f)
    idx_360 = build_360_index(threesixty_data)
    print(f"   Loaded {len(threesixty_data)} 360 frames, indexed {len(idx_360)} unique events.")
    
    # 2) Detect anchor events
    print("\n🔍 Detecting anchor events...")
    anchors = detect_all_anchors(events, idx_360)
    print(f"   Found {len(anchors)} anchor events:")
    for a in anchors:
        evt = a["event"]
        etype = get_event_type(evt)
        player = get_player_last_name(evt)
        minute = evt.get("minute", "?")
        second = evt.get("second", "?")
        print(f"      [{minute}:{second:02d}] {etype} by {player} — {a['reason']}")
    
    # 3) Build flattened streams and classify intensity
    print("\n📝 Building flattened event streams...")
    prompts_data = []
    
    for a in anchors:
        anchor_idx = a["index"]
        anchor_event = a["event"]
        
        # Build context window
        context_events = build_context_window(events, anchor_idx, a)
        
        # Get 360 frame for intensity
        frame_360 = idx_360.get(anchor_event.get("id"))
        
        # Classify intensity
        intensity = classify_intensity(anchor_event, frame_360)
        
        # Build the flattened stream
        flattened = build_flattened_stream(context_events, anchor_event, a, idx_360)
        
        # Build prompt
        prompt = build_prompt(flattened, intensity, anchor_event)
        
        prompts_data.append({
            "anchor_index": anchor_idx,
            "anchor_event_id": anchor_event.get("id"),
            "anchor_event_type": get_event_type(anchor_event),
            "player": get_player_last_name(anchor_event),
            "timestamp": anchor_event.get("timestamp"),
            "minute": anchor_event.get("minute"),
            "second": anchor_event.get("second"),
            "reason": a["reason"],
            "intensity": intensity,
            "flattened_stream": flattened,
            "full_prompt": prompt,
        })
    
    print(f"   Generated {len(prompts_data)} prompts.")
    
    # 4) Check Ollama only for key events that use LLM
    needs_llm = any(is_llm_commentary_event(events[pd["anchor_index"]]) for pd in prompts_data)
    llm_available = False

    if needs_llm:
        print("\n🤖 Checking Ollama...")
        if not check_ollama_running():
            print("⚠️  Ollama is not running. Falling back to template commentary.")
        else:
            if not check_model_exists(MODEL_NAME):
                print(f"   Model '{MODEL_NAME}' not found. Creating...")
                if create_model(MODEL_NAME, modelfile_path):
                    llm_available = True
                else:
                    print("⚠️  Failed to create model. Falling back to template commentary.")
            else:
                print(f"   ✅ Model '{MODEL_NAME}' is ready.")
                llm_available = True
    else:
        print("\n🤖 Skipping Ollama checks (selected anchors are template-driven).")
    
    # 5) Generate commentary
    print("\n🎙️  Generating commentary...\n")
    print("-" * 60)
    commentary_output: list[dict] = []
    silent_count = 0
    
    for i, pd_entry in enumerate(prompts_data):
        minute = pd_entry["minute"]
        second = pd_entry["second"]
        etype = pd_entry["anchor_event_type"]
        player = pd_entry["player"]
        intensity = pd_entry["intensity"]
        reason = pd_entry["reason"]
        
        print(f"\n[{i+1}/{len(prompts_data)}] ⏱ {minute}:{second:02d} | "
              f"{etype} by {player} | {intensity} | {reason}")

        anchor_idx = pd_entry["anchor_index"]
        anchor_event = events[anchor_idx]
        use_llm = is_llm_commentary_event(anchor_event) and llm_available

        if use_llm:
            commentary = generate_commentary(pd_entry["full_prompt"])
        else:
            commentary = generate_template_commentary(events, anchor_idx, {"reason": reason})

        commentary = ensure_allowed_commentary_names(
            commentary,
            events,
            anchor_idx,
            {"reason": reason},
        )
        if not commentary:
            silent_count += 1
            print("   🔇  (silent)")
            continue

        pd_entry["generated_commentary"] = commentary
        pd_entry["generation_mode"] = "llm" if use_llm else "template"
        commentary_output.append(pd_entry)
        print(f"   🎙️  {commentary}")
    
    print("\n" + "-" * 60)
    print(f"   Final voiced events: {len(commentary_output)} (silent skipped: {silent_count})")
    
    # 6) Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(script_dir, f"commentary_output_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(commentary_output, f, indent=4, ensure_ascii=False)
    print(f"\n✅ Saved commentary to: {output_path}")
    
    # Also save the prompts separately for reference
    prompts_path = os.path.join(script_dir, "generated_prompts.json")
    prompts_only = [{k: v for k, v in pd_entry.items()
                     if k not in ("generated_commentary", "generation_mode")}
                    for pd_entry in prompts_data]
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(prompts_only, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved prompts reference to: {prompts_path}")
    
    print("\n🏁 Pipeline complete!")


if __name__ == "__main__":
    main()
