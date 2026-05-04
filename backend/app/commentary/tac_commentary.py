"""
Commentary Generation Pipeline
===============================
Reads StatsBomb event + 360 JSON files, generates tactical descriptions
using heuristic analysis, and feeds them to the finetuned LLM via Ollama
to produce match commentary.

Usage:
    python generate_commentary.py
"""

import copy
import json
import math
import os
import sys
from datetime import datetime

import numpy as np
import requests

# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# CONFIG
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
EVENT_FILE = "demo1_event.json"
THREESIXTY_FILE = "demo1_threesixty.json"

PITCH_X = 120.0
PITCH_Y = 80.0
PITCH_CENTER = (PITCH_X / 2.0, PITCH_Y / 2.0)

# Selection constraints
SHORT_SEQUENCE_SECONDS = 15.0
SET_PIECE_LOOKBACK_SECONDS = 5.0
MIDFIELD_MAX_CENTER_DISTANCE = 25.0
SLOW_PLAY_QUANTILE = 0.60
PRE_GOAL_SPATIAL_SUPPRESS_SECONDS = 15.0

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "tac-commentary"

SYSTEM_PROMPT = (
    "You are a professional football commentator. "
    "Generate exactly one commentary in the requested level. "
    "Beginner = simple and clear. "
    "Intermediate = moderate tactical detail. "
    "Expert = precise tactical language with natural use of metrics when relevant."
)

LEVELS = ["Beginner", "Intermediate", "Expert"]


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# HELPER FUNCTIONS  (from Tactical_description.ipynb)
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def get_event_type(event):
    if not isinstance(event, dict):
        return "Unknown"
    t = event.get("type")
    if isinstance(t, dict):
        return t.get("name", "Unknown")
    return event.get("type_name", "Unknown")


def get_event_id(event):
    if not isinstance(event, dict):
        return None
    return event.get("id") or event.get("event_uuid") or event.get("event_id")


def get_event_location(event):
    if not isinstance(event, dict):
        return None
    loc = event.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        return [float(loc[0]), float(loc[1])]
    return None


def get_player_name(event):
    if not isinstance(event, dict):
        return None
    val = event.get("player")
    if isinstance(val, dict):
        return val.get("name")
    return val


def get_team_name(event):
    if not isinstance(event, dict):
        return None
    val = event.get("team")
    if isinstance(val, dict):
        return val.get("name")
    return val


def get_pass_recipient(event):
    if not isinstance(event, dict):
        return None
    p = event.get("pass")
    if isinstance(p, dict):
        r = p.get("recipient")
        if isinstance(r, dict):
            return r.get("name")
    return None


def get_shot_outcome(event):
    if not isinstance(event, dict):
        return None
    s = event.get("shot")
    if isinstance(s, dict):
        outcome = s.get("outcome")
        if isinstance(outcome, dict):
            return outcome.get("name")
    return None


def is_goal_event(event):
    et = get_event_type(event)
    if et == "Goal":
        return True
    return et == "Shot" and get_shot_outcome(event) == "Goal"


def is_foul_event(event):
    return get_event_type(event) in {"Foul Won", "Foul Committed", "Foul"}


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# FOCUS EVENT / 360 SELECTION
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def get_event_clock_seconds(event):
    if not isinstance(event, dict):
        return None

    timestamp = event.get("timestamp")
    if isinstance(timestamp, str):
        parts = timestamp.split(":")
        if len(parts) == 3:
            try:
                hh = int(parts[0])
                mm = int(parts[1])
                ss = float(parts[2])
                return (hh * 3600.0) + (mm * 60.0) + ss
            except ValueError:
                pass

    minute = event.get("minute")
    second = event.get("second")
    try:
        if minute is not None and second is not None:
            return (float(minute) * 60.0) + float(second)
    except (TypeError, ValueError):
        return None
    return None


def get_clip_duration_seconds(events):
    event_times = []
    for ev in events:
        ts = get_event_clock_seconds(ev)
        if ts is not None:
            event_times.append(ts)
    if len(event_times) < 2:
        return 0.0
    return float(max(event_times) - min(event_times))


def get_goal_event_times(events):
    goal_times = []
    for ev in events:
        if not is_goal_event(ev):
            continue
        ts = get_event_clock_seconds(ev)
        if ts is None:
            continue
        goal_times.append(float(ts))
    return goal_times


def is_event_within_pre_goal_window(event, goal_times, window_seconds=PRE_GOAL_SPATIAL_SUPPRESS_SECONDS):
    ts = get_event_clock_seconds(event)
    if ts is None:
        return False
    return any(0 < (goal_ts - ts) <= window_seconds for goal_ts in goal_times)


def get_sequence_positive_time_gaps(events):
    timestamps = []
    for ev in events:
        ts = get_event_clock_seconds(ev)
        if ts is not None:
            timestamps.append(ts)

    gaps = []
    for i in range(1, len(timestamps)):
        gap = timestamps[i] - timestamps[i - 1]
        if gap > 0:
            gaps.append(float(gap))
    return gaps


def get_event_local_tempo_gap(events, event_idx):
    if event_idx < 0 or event_idx >= len(events):
        return 0.0

    current_ts = get_event_clock_seconds(events[event_idx])
    if current_ts is None:
        return 0.0

    prev_gap = None
    for i in range(event_idx - 1, -1, -1):
        ts = get_event_clock_seconds(events[i])
        if ts is None:
            continue
        gap = current_ts - ts
        if gap > 0:
            prev_gap = float(gap)
            break

    next_gap = None
    for i in range(event_idx + 1, len(events)):
        ts = get_event_clock_seconds(events[i])
        if ts is None:
            continue
        gap = ts - current_ts
        if gap > 0:
            next_gap = float(gap)
            break

    if prev_gap is None and next_gap is None:
        return 0.0
    if prev_gap is None:
        return next_gap
    if next_gap is None:
        return prev_gap
    return max(prev_gap, next_gap)


def freeze_frame_player_count(freeze_frame_data):
    if not freeze_frame_data or "freeze_frame" not in freeze_frame_data:
        return 0
    count = 0
    for p in freeze_frame_data["freeze_frame"]:
        if not isinstance(p, dict):
            continue
        loc = p.get("location")
        if isinstance(loc, list) and len(loc) >= 2:
            count += 1
    return count


def event_distance_to_pitch_center(event):
    loc = get_event_location(event)
    if loc is None:
        return float("inf")
    return math.dist(loc, PITCH_CENTER)


def is_penalty_or_goal_area_location(loc):
    if not loc or len(loc) < 2:
        return False
    x = float(loc[0])
    y = float(loc[1])

    in_left_penalty = x <= 18.0 and 18.0 <= y <= 62.0
    in_right_penalty = x >= 102.0 and 18.0 <= y <= 62.0
    in_left_goal = x <= 6.0 and 30.0 <= y <= 50.0
    in_right_goal = x >= 114.0 and 30.0 <= y <= 50.0
    return in_left_penalty or in_right_penalty or in_left_goal or in_right_goal


def is_corner_or_free_kick_event(event):
    if not isinstance(event, dict):
        return False

    play_pattern_name = ((event.get("play_pattern") or {}).get("name") or "").lower()
    if "corner" in play_pattern_name or "free kick" in play_pattern_name:
        return True

    pass_type_name = ((event.get("pass") or {}).get("type") or {}).get("name")
    if isinstance(pass_type_name, str) and pass_type_name in {"Corner", "Free Kick"}:
        return True

    shot_type_name = ((event.get("shot") or {}).get("type") or {}).get("name")
    if isinstance(shot_type_name, str) and ("Corner" in shot_type_name or "Free Kick" in shot_type_name):
        return True

    return False


def is_possession_event(event):
    if not isinstance(event, dict):
        return False
    team = event.get("team") or {}
    poss_team = event.get("possession_team") or {}

    team_id = team.get("id") if isinstance(team, dict) else None
    poss_id = poss_team.get("id") if isinstance(poss_team, dict) else None
    if team_id is not None and poss_id is not None:
        return team_id == poss_id

    team_name = team.get("name") if isinstance(team, dict) else team
    poss_name = poss_team.get("name") if isinstance(poss_team, dict) else poss_team
    if team_name and poss_name:
        return team_name == poss_name

    return True


def is_valid_spatial_event_candidate(event, threesixty_lookup, excluded_event_ids=None):
    excluded_event_ids = excluded_event_ids or set()
    event_id = get_event_id(event)
    if not event_id or event_id in excluded_event_ids:
        return False

    if is_goal_event(event) or is_foul_event(event):
        return False

    if not is_possession_event(event):
        return False

    event_loc = get_event_location(event)
    if event_loc is None or is_penalty_or_goal_area_location(event_loc):
        return False

    # Keep spatial snapshot anchored in a midfield zone, except for
    # set-piece triggers that are handled by the 5-second lookback rule.
    if (
        event_distance_to_pitch_center(event) > MIDFIELD_MAX_CENTER_DISTANCE
        and not is_corner_or_free_kick_event(event)
    ):
        return False

    freeze_frame_data = threesixty_lookup.get(event_id)
    if freeze_frame_player_count(freeze_frame_data) <= 0:
        return False

    return True


def rank_spatial_event_candidates(events, threesixty_lookup, excluded_event_ids=None):
    excluded_event_ids = excluded_event_ids or set()
    sequence_gaps = get_sequence_positive_time_gaps(events)
    baseline_gap = float(np.median(sequence_gaps)) if sequence_gaps else 1.0
    baseline_gap = max(baseline_gap, 1e-6)

    ranked = []
    for idx, ev in enumerate(events):
        if not is_valid_spatial_event_candidate(ev, threesixty_lookup, excluded_event_ids):
            continue
        event_id = get_event_id(ev)
        frame = threesixty_lookup.get(event_id)
        player_count = freeze_frame_player_count(frame)
        center_distance = event_distance_to_pitch_center(ev)
        local_gap = get_event_local_tempo_gap(events, idx)
        slow_ratio = local_gap / baseline_gap
        ranked.append(
            {
                "idx": idx,
                "local_gap": float(local_gap),
                "slow_ratio": float(slow_ratio),
                "player_count": int(player_count),
                "center_distance": float(center_distance),
            }
        )

    if not ranked:
        return []

    slow_values = [item["slow_ratio"] for item in ranked]
    slow_threshold = float(np.quantile(slow_values, SLOW_PLAY_QUANTILE))
    slow_candidates = [item for item in ranked if item["slow_ratio"] >= slow_threshold]
    if not slow_candidates:
        slow_candidates = ranked

    slow_candidates.sort(
        key=lambda item: (
            -item["slow_ratio"],         # 1) slower relative play first
            -item["player_count"],       # 2) denser freeze frame next
            item["center_distance"],     # 3) closer to middle
            item["idx"],
        )
    )
    return [item["idx"] for item in slow_candidates]


def choose_pre_set_piece_spatial_index(
    events,
    threesixty_lookup,
    set_piece_idx,
    excluded_event_ids=None,
    target_seconds=SET_PIECE_LOOKBACK_SECONDS,
):
    excluded_event_ids = excluded_event_ids or set()
    set_piece_time = get_event_clock_seconds(events[set_piece_idx])
    if set_piece_time is None:
        return None

    best = None
    for idx in range(set_piece_idx - 1, -1, -1):
        candidate = events[idx]
        candidate_time = get_event_clock_seconds(candidate)
        if candidate_time is None:
            continue

        delta = set_piece_time - candidate_time
        if delta <= 0:
            continue
        if delta > target_seconds + 5.0:
            continue

        if not is_valid_spatial_event_candidate(candidate, threesixty_lookup, excluded_event_ids):
            continue

        frame = threesixty_lookup.get(get_event_id(candidate))
        score = (
            abs(delta - target_seconds),
            event_distance_to_pitch_center(candidate),
            -freeze_frame_player_count(frame),
            -idx,
        )
        if best is None or score < best[0]:
            best = (score, idx)

    return best[1] if best else None


def build_commentary_plan(events, threesixty_lookup):
    plan = []
    excluded_ids = set()
    has_goal = False
    goal_times = get_goal_event_times(events)

    # Mandatory events: all goals and fouls.
    for ev in events:
        event_id = get_event_id(ev)
        if not event_id:
            continue
        if is_goal_event(ev):
            has_goal = True
            plan.append({"event_id": event_id, "selection_reason": "goal"})
            excluded_ids.add(event_id)
        elif is_foul_event(ev):
            plan.append({"event_id": event_id, "selection_reason": "foul"})
            excluded_ids.add(event_id)

    # Optional one-off spatial analysis event.
    sequence_duration = get_clip_duration_seconds(events)
    skip_spatial = has_goal and sequence_duration <= SHORT_SEQUENCE_SECONDS
    if not skip_spatial:
        ranked_candidates = rank_spatial_event_candidates(events, threesixty_lookup, excluded_ids)
        for idx in ranked_candidates:
            candidate = events[idx]
            candidate_id = get_event_id(candidate)
            selected_event = candidate
            selected_id = candidate_id
            selection_reason = "spatial_midfield_dense_frame"
            source_event_id = candidate_id

            if is_corner_or_free_kick_event(candidate):
                prior_idx = choose_pre_set_piece_spatial_index(
                    events, threesixty_lookup, idx, excluded_ids
                )
                if prior_idx is None:
                    continue
                selected_event = events[prior_idx]
                selected_id = get_event_id(selected_event)
                selection_reason = "spatial_5s_before_set_piece"

            if selected_event and is_event_within_pre_goal_window(selected_event, goal_times):
                continue

            if selected_id and selected_id not in excluded_ids:
                plan.append(
                    {
                        "event_id": selected_id,
                        "selection_reason": selection_reason,
                        "source_event_id": source_event_id,
                    }
                )
                excluded_ids.add(selected_id)
                break

    # Keep final processing in event timeline order.
    id_to_index = {}
    for i, ev in enumerate(events):
        eid = get_event_id(ev)
        if eid and eid not in id_to_index:
            id_to_index[eid] = i
    plan.sort(key=lambda item: id_to_index.get(item["event_id"], 10 ** 9))
    return plan


def choose_focus_event_index(events):
    if not events:
        return None
    goal_indices = [i for i, ev in enumerate(events) if is_goal_event(ev)]
    if goal_indices:
        return goal_indices[-1]
    foul_indices = [i for i, ev in enumerate(events) if is_foul_event(ev)]
    if foul_indices:
        return foul_indices[-1]
    return len(events) - 1


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# SEQUENCE SUMMARY
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def action_phrase(event):
    et = get_event_type(event)
    player = get_player_name(event) or "A player"

    if et == "Pass":
        recipient = get_pass_recipient(event)
        if recipient:
            return f"{player} passes to {recipient}"
        return f"{player} plays a pass"
    if et == "Carry":
        return f"{player} carries the ball"
    if et == "Dribble":
        return f"{player} dribbles forward"
    if et == "Shot":
        if get_shot_outcome(event) == "Goal":
            return f"{player} finishes the move"
        return f"{player} takes the shot"
    if et == "Foul Won":
        return f"{player} draws the foul"
    if et == "Foul Committed":
        return f"{player} commits the foul"
    if et == "Duel":
        return f"{player} contests the duel"
    return f"{player} performs the action"


def summarize_sequence(events, focus_idx, max_buildup_events=4):
    if not events or focus_idx is None or focus_idx <= 0:
        return ""
    start_idx = max(0, focus_idx - max_buildup_events)
    buildup = events[start_idx:focus_idx]
    if not buildup:
        return ""
    phrases = [action_phrase(ev) for ev in buildup]
    focus_phrase = action_phrase(events[focus_idx])
    if len(phrases) == 1:
        return f"The move develops as {phrases[0]}, before {focus_phrase}."
    if len(phrases) == 2:
        return f"The move develops as {phrases[0]}, then {phrases[1]}, before {focus_phrase}."
    middle = ", then ".join(phrases[:-1])
    return f"The move develops as {middle}, then {phrases[-1]}, before {focus_phrase}."


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# DIRECTION / NORMALIZATION
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def infer_attacking_direction_right(freeze_frame_data):
    if not freeze_frame_data or "freeze_frame" not in freeze_frame_data:
        return True
    teammates_gk_x = []
    opponents_gk_x = []
    for p in freeze_frame_data["freeze_frame"]:
        if not isinstance(p, dict):
            continue
        loc = p.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        x = float(loc[0])
        teammate = bool(p.get("teammate", False))
        keeper = bool(p.get("keeper", False))
        if keeper and teammate:
            teammates_gk_x.append(x)
        elif keeper and not teammate:
            opponents_gk_x.append(x)
    if teammates_gk_x and opponents_gk_x:
        return np.mean(opponents_gk_x) > np.mean(teammates_gk_x)
    if opponents_gk_x:
        return np.mean(opponents_gk_x) > 60.0
    if teammates_gk_x:
        return np.mean(teammates_gk_x) < 60.0
    return True


def normalize_x(x, attacking_right=True):
    return x if attacking_right else (PITCH_X - x)


def normalize_location(loc, attacking_right=True):
    if not loc or len(loc) < 2:
        return None
    x = normalize_x(float(loc[0]), attacking_right)
    y = float(loc[1])
    return [x, y]


def normalize_freeze_frame(freeze_frame_data, attacking_right=True):
    if not freeze_frame_data or "freeze_frame" not in freeze_frame_data:
        return None
    ff_copy = copy.deepcopy(freeze_frame_data)
    new_ff = []
    for p in ff_copy["freeze_frame"]:
        if not isinstance(p, dict):
            continue
        loc = p.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        p["location"] = normalize_location(loc, attacking_right)
        new_ff.append(p)
    ff_copy["freeze_frame"] = new_ff
    return ff_copy


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# TACTICAL ANALYSIS (HEURISTIC)
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def split_freeze_frame(freeze_frame_data):
    teammates, opponents = [], []
    if not freeze_frame_data or "freeze_frame" not in freeze_frame_data:
        return np.empty((0, 2)), np.empty((0, 2))
    for p in freeze_frame_data["freeze_frame"]:
        loc = p.get("location")
        if not isinstance(loc, list) or len(loc) < 2:
            continue
        xy = [float(loc[0]), float(loc[1])]
        if p.get("teammate", False):
            teammates.append(xy)
        else:
            opponents.append(xy)
    return (
        np.array(teammates) if teammates else np.empty((0, 2)),
        np.array(opponents) if opponents else np.empty((0, 2)),
    )


def spread_metrics(coords):
    if len(coords) == 0:
        return {"width": np.nan, "depth": np.nan, "mean_x": np.nan, "std_y": np.nan}
    x, y = coords[:, 0], coords[:, 1]
    return {
        "width": float(y.max() - y.min()),
        "depth": float(x.max() - x.min()),
        "mean_x": float(x.mean()),
        "std_y": float(y.std()),
    }


def infer_formation_approx(teammates):
    if len(teammates) < 4:
        return "Unclear"
    x = teammates[:, 0]
    deep_players = int((x < 55).sum())
    if deep_players >= 5:
        return "Back Five Approx"
    elif deep_players == 4:
        return "Back Four Approx"
    elif deep_players == 3:
        return "Back Three Approx"
    else:
        return "Unclear"


def analyze_tactical_snapshot(event_data, freeze_frame_data):
    result = {
        "team_shape": "Unknown",
        "formation_approx": "Unclear",
        "attacking_structure": "Unknown",
        "defensive_block": "Unknown",
        "defensive_shape": "Unknown",
    }
    event_xy = get_event_location(event_data)
    teammates, opponents = split_freeze_frame(freeze_frame_data)
    if event_xy is None or len(teammates) == 0 or len(opponents) == 0:
        return result

    team_metrics = spread_metrics(teammates)
    opp_metrics = spread_metrics(opponents)

    # Team shape
    if team_metrics["width"] >= 42 and team_metrics["depth"] >= 32:
        result["team_shape"] = "Stretched Shape"
    elif team_metrics["width"] >= 42:
        result["team_shape"] = "Wide Shape"
    elif team_metrics["depth"] >= 32:
        result["team_shape"] = "Vertical Shape"
    else:
        result["team_shape"] = "Compact Shape"

    # Attacking structure
    if team_metrics["width"] >= 45:
        result["attacking_structure"] = "Wide Structure"
    elif team_metrics["width"] <= 28 and team_metrics["std_y"] <= 12:
        result["attacking_structure"] = "Central Overload"
    elif team_metrics["depth"] >= 35:
        result["attacking_structure"] = "Vertical Support Structure"
    else:
        result["attacking_structure"] = "Balanced Structure"

    # Defensive block
    if opp_metrics["mean_x"] >= 92:
        result["defensive_block"] = "Low Block"
    elif opp_metrics["mean_x"] >= 68:
        result["defensive_block"] = "Mid Block"
    else:
        result["defensive_block"] = "High Press"

    # Defensive shape
    width_tag = (
        "Narrow" if opp_metrics["width"] < 28
        else "Wide" if opp_metrics["width"] > 42
        else "Balanced"
    )
    compactness_tag = "Compact" if opp_metrics["depth"] < 24 else "Spread"
    result["defensive_shape"] = f"{compactness_tag} {width_tag} {result['defensive_block']}"
    result["formation_approx"] = infer_formation_approx(teammates)

    return result


def compute_support_and_opposition_context(event_data, freeze_frame_data):
    result = {
        "support_context": "Support context unavailable.",
        "opposition_effect": "Opposition effect unavailable.",
    }
    event_xy = get_event_location(event_data)
    teammates, opponents = split_freeze_frame(freeze_frame_data)
    if event_xy is None or len(teammates) == 0 or len(opponents) == 0:
        return result

    evt = np.array(event_xy)
    team_dists = np.linalg.norm(teammates - evt, axis=1)
    opp_dists = np.linalg.norm(opponents - evt, axis=1)

    close_support = int((team_dists <= 12).sum())
    close_opponents = int((opp_dists <= 8).sum())

    central_lane_blocked = int(
        ((opponents[:, 0] > evt[0]) & (np.abs(opponents[:, 1] - evt[1]) <= 10)).sum()
    ) > 0

    wide_option_available = int(
        ((teammates[:, 0] >= evt[0] - 5) & (np.abs(teammates[:, 1] - evt[1]) >= 15)).sum()
    ) > 0

    # Support text
    if close_support >= 3:
        support_text = "Several nearby teammates provide close support around the action."
    elif close_support == 2:
        support_text = "There is immediate supporting presence around the action."
    elif close_support == 1:
        support_text = "Support is limited to a single nearby option."
    else:
        support_text = "The player appears relatively isolated at the moment of the action."

    if wide_option_available:
        support_text += " A wider outlet is also available."

    # Opposition effect text
    if close_opponents >= 3 and central_lane_blocked:
        opp_text = "The opposition compress space around the action and block central progression."
    elif central_lane_blocked:
        opp_text = "The opposition shape protects the central lane and encourages play away from the middle."
    elif close_opponents >= 2:
        opp_text = "The opposition apply local pressure around the action without fully sealing central access."
    else:
        opp_text = "The opposition shape is present but leaves some room around the action."

    result["support_context"] = support_text
    result["opposition_effect"] = opp_text
    return result


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# TACTICAL DESCRIPTION COMPOSITION
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def compose_tactical_description(team, player, sequence_summary,
                                  gnn_pred, opposition_effect,
                                  support_context):
    """
    Mirrors compose_tactical_description from the notebook.
    gnn_pred here is the heuristic labels dict (same structure).
    No pressure / metric context is included.
    """
    team = team or "The attacking side"
    player = player or "The player"

    team_shape      = gnn_pred["team_shape"]
    formation_approx = gnn_pred["formation_approx"]
    attacking_structure = gnn_pred["attacking_structure"]
    defensive_shape = gnn_pred["defensive_shape"]

    sentences = []

    if sequence_summary:
        sentences.append(sequence_summary)

    sentences.append(
        f"{team} are set in a {formation_approx.lower()} with a "
        f"{attacking_structure.lower()} and a {team_shape.lower()}."
    )

    sentences.append(
        f"The opposition defend in a {defensive_shape.lower()}."
    )

    if opposition_effect:
        sentences.append(opposition_effect)

    if support_context:
        sentences.append(support_context)

    return " ".join(sentences)


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# OLLAMA LLM CALL
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def spatial_commentary_has_required_points(commentary, tactical_labels):
    if not commentary:
        return False
    text = commentary.lower()
    required_values = [
        str(tactical_labels.get("team_shape", "")).lower(),
        str(tactical_labels.get("formation_approx", "")).lower(),
        str(tactical_labels.get("attacking_structure", "")).lower(),
        str(tactical_labels.get("defensive_shape", "")).lower(),
    ]
    required_values = [val for val in required_values if val]
    return all(val in text for val in required_values)


def formation_approx_to_phrase(formation_approx):
    mapping = {
        "Back Five Approx": "back-five base",
        "Back Four Approx": "back-four base",
        "Back Three Approx": "back-three base",
        "Unclear": "fluid base",
    }
    return mapping.get(formation_approx, str(formation_approx).lower())


def attacking_structure_to_formation(attacking_structure):
    mapping = {
        "Wide Structure": "wide attacking formation",
        "Central Overload": "central overload formation",
        "Vertical Support Structure": "vertical support formation",
        "Balanced Structure": "balanced attacking formation",
    }
    return mapping.get(attacking_structure, str(attacking_structure).lower())


def build_spatial_commentary_fallback(level, tactical_labels, team_name=None):
    team_name = team_name or "The possession side"
    team_shape = tactical_labels.get("team_shape", "Unknown")
    formation = tactical_labels.get("formation_approx", "Unclear")
    attacking_structure = tactical_labels.get("attacking_structure", "Unknown")
    defensive_shape = tactical_labels.get("defensive_shape", "Unknown")
    formation_phrase = formation_approx_to_phrase(formation)
    attacking_formation = attacking_structure_to_formation(attacking_structure)

    if level == "Beginner":
        return (
            f"{team_name} hold a {team_shape.lower()} here. "
            f"They look set in a {formation_phrase}, attacking in a {attacking_formation}. "
            f"The opposition are organized in a {defensive_shape.lower()}."
        )
    if level == "Intermediate":
        return (
            f"{team_name} are in a {team_shape.lower()} with a {formation_phrase}. "
            f"They attack through a {attacking_formation}, while the opposition keep a "
            f"{defensive_shape.lower()}."
        )
    return (
        f"{team_name} show a {formation_phrase} and a {team_shape.lower()} in this slower phase. "
        f"Their attack takes a {attacking_formation}, and the out-of-possession unit hold a "
        f"{defensive_shape.lower()} to protect central access."
    )


def generate_commentary_ollama(
    tactical_desc,
    level,
    selection_reason="auto",
    tactical_labels=None,
    team_name=None,
    analytics_context=None,
    short_single_sentence=False,
):
    """Call the Ollama API and return the generated commentary."""
    is_spatial_snapshot = str(selection_reason).startswith("spatial_")
    tactical_labels = tactical_labels or {}
    
    metric_str = f"Metric Context:\n{analytics_context}" if analytics_context else "Metric Context: Stats are not available for this action."
    short_clip_instruction = (
        "\n\nShort-clip exception: return exactly one short sentence only. "
        "Keep it punchy, tactical, and under 14 words."
        if short_single_sentence
        else ""
    )

    if is_spatial_snapshot:
        user_content = (
            f"Level: {level}\n\n"
            "Task: Generate commentary for a tactical freeze-frame.\n"
            "Focus only on spatial/team-structure interpretation from this snapshot.\n"
            "Do not invent unrelated actions, players, or outcomes.\n\n"
            f"Possession Team: {team_name or 'The possession side'}\n"
            "Use the possession team name directly in the commentary.\n"
            "Avoid meta phrases like 'this freeze-frame' or 'this snapshot'.\n"
            "Use the term 'attacking formation' (not 'attacking structure').\n\n"
            "You must explicitly include all 4 points below in natural commentary:\n"
            f"1) Team shape: {tactical_labels.get('team_shape', 'Unknown')}\n"
            f"2) Formation approximation: {tactical_labels.get('formation_approx', 'Unclear')}\n"
            f"3) Attacking formation: {attacking_structure_to_formation(tactical_labels.get('attacking_structure', 'Unknown'))}\n"
            f"4) Defensive structure: {tactical_labels.get('defensive_shape', 'Unknown')}\n\n"
            f"Snapshot Description: {tactical_desc}\n\n"
            f"{metric_str}"
            f"{short_clip_instruction}"
        )
    else:
        user_content = (
            f"Level: {level}\n\n"
            f"Tactical Description: {tactical_desc}\n\n"
            f"{metric_str}"
            f"{short_clip_instruction}"
        )

    system_prompt = SYSTEM_PROMPT
    if short_single_sentence:
        system_prompt = (
            f"{SYSTEM_PROMPT} For short clips, output exactly one short sentence and do not add a second sentence."
        )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        commentary = data.get("message", {}).get("content", "").strip()
        if is_spatial_snapshot and not spatial_commentary_has_required_points(commentary, tactical_labels):
            return build_spatial_commentary_fallback(level, tactical_labels, team_name=team_name)
        return commentary
    except requests.exceptions.ConnectionError:
        if is_spatial_snapshot:
            return build_spatial_commentary_fallback(level, tactical_labels, team_name=team_name)
        return "[ERROR] Could not connect to Ollama. Is it running? (ollama serve)"
    except Exception as e:
        if is_spatial_snapshot:
            return build_spatial_commentary_fallback(level, tactical_labels, team_name=team_name)
        return f"[ERROR] {e}"


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# MAIN PIPELINE
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def process_event(
    event,
    events_list,
    threesixty_lookup,
    selection_reason="auto",
    source_event_id=None,
):
    """
    Given a single target event dict, its containing events list,
    and the 360 lookup, generate the tactical description.
    Returns a dict with event metadata + tactical_description.
    """
    event_id = get_event_id(event)
    event_type = get_event_type(event)
    player = get_player_name(event)
    team = get_team_name(event)

    print(f"\n{'='*70}")
    print(f"  Processing Event: {event_id}")
    print(f"  Type: {event_type} | Player: {player} | Team: {team}")
    print(f"  Selection Reason: {selection_reason}")
    if source_event_id and source_event_id != event_id:
        print(f"  Source Event: {source_event_id}")
    print(f"{'='*70}")

    # ΓöÇΓöÇ Determine if this is a goal / foul event ΓöÇΓöÇ
    goal = is_goal_event(event)
    foul = is_foul_event(event)

    # ΓöÇΓöÇ Build the event sequence (for goal / foul, include related events) ΓöÇΓöÇ
    if goal or foul:
        # Collect related events from the full events list
        related_ids = set(event.get("related_events", []))
        # For shots, also pull in the key_pass_id
        shot_data = event.get("shot", {})
        key_pass_id = shot_data.get("key_pass_id")
        if key_pass_id:
            related_ids.add(key_pass_id)

        # Find indices of related events in events_list
        id_to_idx = {get_event_id(e): i for i, e in enumerate(events_list)}
        related_indices = []
        for rid in related_ids:
            if rid in id_to_idx:
                related_indices.append(id_to_idx[rid])

        # Include the focus event itself
        focus_event_idx = id_to_idx.get(event_id)
        if focus_event_idx is not None:
            related_indices.append(focus_event_idx)

        related_indices = sorted(set(related_indices))
        event_sequence = [events_list[i] for i in related_indices]
    else:
        # Regular event ΓÇö single event in the sequence
        event_sequence = [event]

    # ΓöÇΓöÇ Choose focus event ΓöÇΓöÇ
    focus_idx = choose_focus_event_index(event_sequence)
    focus_event = event_sequence[focus_idx] if focus_idx is not None else event

    # ΓöÇΓöÇ Get 360 data ΓÇö ONLY from threesixty.json ΓöÇΓöÇ
    focus_event_id = get_event_id(focus_event)
    focus_360 = threesixty_lookup.get(focus_event_id)

    # If the focus event doesn't have 360, try the original target event
    if focus_360 is None:
        focus_360 = threesixty_lookup.get(event_id)

    if focus_360 is None:
        print("  [WARNING] No 360 data found for this event in threesixty.json")

    # ΓöÇΓöÇ Normalize direction ΓöÇΓöÇ
    attacking_right = infer_attacking_direction_right(focus_360)

    norm_focus_event = copy.deepcopy(focus_event)
    if norm_focus_event is not None and get_event_location(norm_focus_event) is not None:
        norm_focus_event["location"] = normalize_location(
            get_event_location(norm_focus_event), attacking_right
        )

    norm_focus_360 = normalize_freeze_frame(focus_360, attacking_right)

    # ΓöÇΓöÇ Heuristic tactical labels ΓöÇΓöÇ
    labels = analyze_tactical_snapshot(norm_focus_event, norm_focus_360)
    ctx = compute_support_and_opposition_context(norm_focus_event, norm_focus_360)

    # ΓöÇΓöÇ Sequence summary ΓöÇΓöÇ
    seq_summary = summarize_sequence(event_sequence, focus_idx)

    # ΓöÇΓöÇ Compose tactical description ΓöÇΓöÇ
    tactical_desc = compose_tactical_description(
        team=team,
        player=player,
        sequence_summary=seq_summary,
        gnn_pred=labels,
        opposition_effect=ctx["opposition_effect"],
        support_context=ctx["support_context"],
    )

    print(f"\n  Tactical Description:\n  {tactical_desc}\n")

    return {
        "event_id": event_id,
        "event_type": event_type,
        "player": player,
        "team": team,
        "is_goal": goal,
        "is_foul": foul,
        "selection_reason": selection_reason,
        "source_event_id": source_event_id,
        "tactical_labels": labels,
        "tactical_description": tactical_desc,
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ΓöÇΓöÇ Load data ΓöÇΓöÇ
    event_path = os.path.join(script_dir, EVENT_FILE)
    threesixty_path = os.path.join(script_dir, THREESIXTY_FILE)

    print(f"Loading events from: {event_path}")
    with open(event_path, "r", encoding="utf-8") as f:
        events_list = json.load(f)
    print(f"  Loaded {len(events_list)} events")

    print(f"Loading 360 data from: {threesixty_path}")
    with open(threesixty_path, "r", encoding="utf-8") as f:
        threesixty_list = json.load(f)
    print(f"  Loaded {len(threesixty_list)} 360 frames")

    # ΓöÇΓöÇ Build lookups ΓöÇΓöÇ
    threesixty_lookup = {}
    for frame in threesixty_list:
        uid = frame.get("event_uuid")
        if uid:
            threesixty_lookup[uid] = frame

    event_id_to_event = {}
    for ev in events_list:
        eid = get_event_id(ev)
        if eid:
            event_id_to_event[eid] = ev

    commentary_plan = build_commentary_plan(events_list, threesixty_lookup)

    if not commentary_plan:
        print("\n[WARNING] No eligible events found for commentary generation.")
        return 0

    print(f"\nSelected {len(commentary_plan)} event(s) for commentary generation:")
    for item in commentary_plan:
        print(f"  - {item['event_id']} ({item['selection_reason']})")

    # Process selected events
    all_results = []

    for item in commentary_plan:
        target_id = item["event_id"]
        event = event_id_to_event.get(target_id)
        if event is None:
            print(f"\n[WARNING] Planned event {target_id} not found in {EVENT_FILE}")
            continue

        result = process_event(
            event,
            events_list,
            threesixty_lookup,
            selection_reason=item.get("selection_reason", "auto"),
            source_event_id=item.get("source_event_id"),
        )

        # Generate commentary at each level via Ollama
        commentaries = {}
        for level in LEVELS:
            print(f"  Generating {level} commentary via Ollama...")
            commentary = generate_commentary_ollama(
                result["tactical_description"],
                level,
                selection_reason=result.get("selection_reason", "auto"),
                tactical_labels=result.get("tactical_labels"),
                team_name=result.get("team"),
            )
            commentaries[level] = commentary
            print(f"    Done ({len(commentary)} chars)")

        result["commentaries"] = commentaries
        all_results.append(result)


    # ΓöÇΓöÇ Save output ΓöÇΓöÇ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(script_dir, f"commentary_output_{timestamp}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for res in all_results:
            f.write("=" * 80 + "\n")
            f.write(f"Event ID   : {res['event_id']}\n")
            f.write(f"Event Type : {res['event_type']}\n")
            f.write(f"Player     : {res['player']}\n")
            f.write(f"Team       : {res['team']}\n")
            f.write(f"Is Goal    : {res['is_goal']}\n")
            f.write(f"Is Foul    : {res['is_foul']}\n")
            f.write(f"Selected By: {res['selection_reason']}\n")
            if res.get('source_event_id') and res['source_event_id'] != res['event_id']:
                f.write(f"Source Event: {res['source_event_id']}\n")
            labels = res.get("tactical_labels", {})
            if labels:
                f.write(f"Team Shape : {labels.get('team_shape', 'Unknown')}\n")
                f.write(f"Formation  : {labels.get('formation_approx', 'Unclear')}\n")
                f.write(f"Attack Str.: {labels.get('attacking_structure', 'Unknown')}\n")
                f.write(f"Def. Struct: {labels.get('defensive_shape', 'Unknown')}\n")
            f.write(f"\nTactical Description:\n{res['tactical_description']}\n")
            f.write("\n--- Generated Commentaries ---\n")
            for level in LEVELS:
                f.write(f"\n[{level}]\n")
                f.write(res["commentaries"].get(level, "(no output)") + "\n")
            f.write("\n")

    print(f"\n{'='*70}")
    print(f"  Commentary saved to: {output_file}")
    print(f"{'='*70}")

    # ΓöÇΓöÇ Also print to console ΓöÇΓöÇ
    for res in all_results:
        print(f"\n{'='*70}")
        print(f"  {res['event_type']} by {res['player']} ({res['team']})")
        if res["is_goal"]:
            print("  *** GOAL ***")
        if res["is_foul"]:
            print("  *** FOUL ***")
        print(f"  Selected By: {res['selection_reason']}")
        if res.get("source_event_id") and res["source_event_id"] != res["event_id"]:
            print(f"  Source Event: {res['source_event_id']}")
        print(f"{'='*70}")
        for level in LEVELS:
            print(f"\n  [{level}]")
            print(f"  {res['commentaries'].get(level, '(no output)')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
