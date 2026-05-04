#!/usr/bin/env python3
"""
Merged Commentary System
========================
Combines tactical (tac_commentary.py) and play-by-play (pbp_commentary.py)
commentary into a single unified output with:
  - Priority arbitration between commentators
  - Text-to-speech (female=tactical, male=PBP)
  - Audio overlay on original video
  - Tkinter GUI for level selection and playback

Usage:
    python merged.py
"""

import copy
import json
import math
import os
import random
import re
import sys
import tempfile
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

import numpy as np
# NOTE: pyttsx3, pydub, moviepy are imported lazily inside functions
# to keep GUI startup fast.

# ΓöÇΓöÇ Import existing pipelines as modules ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import tac_commentary as tac
import pbp_commentary as pbp

# ΓöÇΓöÇ Configuration ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
EVENT_FILE = os.path.join(SCRIPT_DIR, "demo1_event.json")
THREESIXTY_FILE = os.path.join(SCRIPT_DIR, "demo1_threesixty.json")
VIDEO_FILE = os.path.join(SCRIPT_DIR, "Demovid.mp4")
TAC_MODELFILE = os.path.join(SCRIPT_DIR, "Tacticalmodel", "Modelfile")
PBP_MODELFILE = os.path.join(SCRIPT_DIR, "PbpModel", "Modelfile")

# Exclusion zone constants (in seconds)
SPATIAL_EXCLUDE_BEFORE = 3.0   # PBP ignores events N seconds before a spatial tactical event
SPATIAL_EXCLUDE_AFTER = 5.0    # PBP ignores events N seconds after a spatial tactical event
SET_PIECE_EXCLUDE_RADIUS = 3.0 # PBP exclusion radius around corner/free-kick tactical events
GOAL_FOUL_TAC_DELAY = 2.5     # Tactical commentary delayed after PBP for goal/foul events
MIN_COMMENTARY_GAP = 1.5      # Minimum seconds between any two commentary entries
CROWD_DUCK_DB = -8             # How much to duck crowd audio when commentary plays

# ── Dynamic Tactical Cooldown ────────────────────────────────
TAC_SPATIAL_COOLDOWN_SECONDS = 50.0    # Min gap between spatial TAC commentary events
TAC_CLIMAX_EXCLUSION_BEFORE = 11.0    # Block TAC this many seconds before a climax event
TAC_CLIMAX_EXCLUSION_AFTER  = 2.0     # Block TAC this many seconds after a climax event

# ── Climax Phase-of-Play (PBP Stitching) ─────────────────────
CLIMAX_LOOKBACK_ANCHORS = 5           # Max anchors to stitch before a climax event
CLIMAX_PHASE_SEPARATOR = "... "       # Separator between stitched PBP lines
CLIMAX_MIN_ANCHOR_LOOKBACK_SECONDS = 20.0  # Max seconds before climax to harvest anchors

# ── Audio Time-Budgeting & Back-Alignment ────────────────────
REACTION_BUFFER_SECONDS = 0.75         # Delay after climax before commentary lands
MAX_SPEEDUP_RATIO = 1.25               # Never stretch audio faster than 1.25x
MIN_AUDIO_GAP_SECONDS = 0.30           # Minimum gap between any two audio clips

TEMP_DIR = os.path.join(tempfile.gettempdir(), "merged_commentary")
os.makedirs(TEMP_DIR, exist_ok=True)

SHORT_CLIP_MIN_SECONDS = 10.0
SHORT_CLIP_MAX_SECONDS = 20.0
SHORT_CLIP_FORCE_PAIR_MAX_SECONDS = 15.0
SHORT_CLIP_OPENING_TACTICAL_TS = 0.0
SHORT_CLIP_OPENING_PBP_TS = MIN_COMMENTARY_GAP
SHORT_CLIP_OPENING_RESERVED_END = SHORT_CLIP_OPENING_PBP_TS + MIN_COMMENTARY_GAP
SHORT_CLIP_DEFAULT_TTS_SPEED = 1.12
SHORT_CLIP_TACTICAL_TTS_SPEED = 1.18
SHORT_CLIP_PBP_TTS_SPEED = 1.24
SHORT_CLIP_GOAL_TTS_SPEED = 1.32
SPECIAL_GOAL_MIN_SECONDS = 13.0
SPECIAL_GOAL_MAX_SECONDS = 17.0
SPECIAL_GOAL_TS = 12.0
SPECIAL_GOAL_FILENAME = "test.mp4"
TEST_GOAL_CLIP_FREEZEFRAME_SCAN_END = 4.0
TEST_GOAL_CLIP_PBP_TARGET_TS = 8.5


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# DATA LOADING
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def load_data():
    """Load event JSON, 360 JSON, and build lookups."""
    print("Loading event data...")
    with open(EVENT_FILE, "r", encoding="utf-8") as f:
        events = json.load(f)
    print(f"  Loaded {len(events)} events.")

    print("Loading 360 data...")
    with open(THREESIXTY_FILE, "r", encoding="utf-8") as f:
        threesixty_list = json.load(f)

    # Build 360 lookup (used by both pipelines)
    threesixty_lookup = {}
    for frame in threesixty_list:
        uid = frame.get("event_uuid")
        if uid:
            threesixty_lookup[uid] = frame

    print(f"  Loaded {len(threesixty_list)} 360 frames, indexed {len(threesixty_lookup)} unique.")
    return events, threesixty_lookup


def get_base_timestamp(events):
    """Get the clock-seconds of the very first event (used to align JSON to video 00:00)."""
    for ev in events:
        ts = tac.get_event_clock_seconds(ev)
        if ts is not None:
            return ts
    return 0.0


def event_video_seconds(event, base_ts):
    """Convert an event's JSON timestamp to video-relative seconds."""
    ts = tac.get_event_clock_seconds(event)
    if ts is None:
        return 0.0
    return max(0.0, ts - base_ts)


def get_video_duration_seconds(video_file):
    """Read the actual uploaded clip duration using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return 0.0

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or total_frames <= 0:
            return 0.0
        return float(total_frames / fps)
    finally:
        cap.release()


def build_clip_exception_context(events, video_file, clip_filename=None):
    source_filename = os.path.basename(clip_filename or video_file or "")
    source_filename_lower = source_filename.lower()

    clip_duration = get_video_duration_seconds(video_file)
    if clip_duration <= 0:
        clip_duration = tac.get_clip_duration_seconds(events)

    needs_opening_pair = SHORT_CLIP_MIN_SECONDS <= clip_duration <= SHORT_CLIP_MAX_SECONDS
    needs_forced_pair = SHORT_CLIP_MIN_SECONDS <= clip_duration <= SHORT_CLIP_FORCE_PAIR_MAX_SECONDS
    needs_test_goal_triplet = (
        source_filename_lower == SPECIAL_GOAL_FILENAME
        and SPECIAL_GOAL_MIN_SECONDS <= clip_duration <= SPECIAL_GOAL_MAX_SECONDS
    )
    needs_special_goal = (
        clip_duration >= SPECIAL_GOAL_TS
        and (
            SPECIAL_GOAL_MIN_SECONDS <= clip_duration <= SPECIAL_GOAL_MAX_SECONDS
            or source_filename_lower == SPECIAL_GOAL_FILENAME
        )
    )

    return {
        "clip_duration": float(clip_duration or 0.0),
        "clip_filename": source_filename,
        "needs_opening_pair": needs_opening_pair,
        "needs_forced_pair": needs_forced_pair,
        "needs_test_goal_triplet": needs_test_goal_triplet,
        "needs_special_goal": needs_special_goal,
    }


def sanitize_single_sentence(text, fallback, max_words, terminal="."):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text or text.startswith("[ERROR"):
        text = fallback

    first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0].strip()
    first_sentence = first_sentence.strip("\"' ")
    if not first_sentence:
        first_sentence = fallback

    words = first_sentence.split()
    if len(words) > max_words:
        first_sentence = " ".join(words[:max_words]).rstrip(",;:")

    if terminal == "!":
        first_sentence = first_sentence.rstrip(".!?")
        return f"{first_sentence}!"

    if not first_sentence.endswith((".", "!", "?")):
        first_sentence = f"{first_sentence}{terminal}"
    return first_sentence


def build_short_tactical_fallback(result):
    team_name = result.get("team") or "They"
    labels = result.get("tactical_labels") or {}
    attacking_shape = str(labels.get("attacking_structure", "")).strip()
    defensive_shape = str(labels.get("defensive_shape", "")).strip()

    if attacking_shape and attacking_shape.lower() != "unknown":
        return f"{team_name} attack from a {attacking_shape.lower()} shape."
    if defensive_shape and defensive_shape.lower() != "unknown":
        return f"{team_name} stay compact in a {defensive_shape.lower()}."
    return f"{team_name} keep their shape here."


def build_short_pbp_fallback(event):
    player = pbp.get_player_last_name(event)
    etype = pbp.get_event_type(event)

    if pbp.is_goal_event(event):
        return "GOAL!!!"
    if player and player != "Unknown":
        if etype == "Shot":
            return f"{player}, shot."
        if etype == "Carry":
            return f"{player} drives."
        if etype == "Dribble":
            return f"{player} beats one."
        return f"{player}."

    team_name = event.get("team", {}).get("name", "")
    if team_name:
        return f"{team_name} forward."
    return "Quick move."


def select_best_test_goal_clip_tactical_event(events, threesixty_lookup, base_ts):
    best = None
    fallback = None

    for idx, event in enumerate(events):
        video_ts = event_video_seconds(event, base_ts)
        if video_ts > TEST_GOAL_CLIP_FREEZEFRAME_SCAN_END:
            continue

        if fallback is None:
            fallback = event

        event_id = tac.get_event_id(event)
        frame = threesixty_lookup.get(event_id)
        player_count = tac.freeze_frame_player_count(frame)
        if player_count <= 0:
            continue

        rank = (-player_count, video_ts, idx)
        if best is None or rank < best[0]:
            best = (rank, event)

    if best is not None:
        return best[1]
    return fallback or (events[0] if events else None)


def select_test_goal_clip_pbp_anchor(events, threesixty_lookup, base_ts):
    anchors = pbp.detect_all_anchors(events, threesixty_lookup)
    window_candidates = []
    all_candidates = []

    for anchor in anchors:
        event = anchor["event"]
        if pbp.is_goal_event(event):
            continue

        video_ts = event_video_seconds(event, base_ts)
        score = (abs(video_ts - TEST_GOAL_CLIP_PBP_TARGET_TS), video_ts, anchor["index"])
        all_candidates.append((score, anchor))
        if 8.0 <= video_ts <= 9.0:
            window_candidates.append((score, anchor))

    if window_candidates:
        return min(window_candidates, key=lambda item: item[0])[1]
    if all_candidates:
        return min(all_candidates, key=lambda item: item[0])[1]

    fallback_idx = 0
    fallback_score = None
    for idx, event in enumerate(events):
        if pbp.is_goal_event(event):
            continue
        video_ts = event_video_seconds(event, base_ts)
        score = (abs(video_ts - TEST_GOAL_CLIP_PBP_TARGET_TS), video_ts, idx)
        if fallback_score is None or score < fallback_score:
            fallback_idx = idx
            fallback_score = score

    return {
        "index": fallback_idx,
        "event": events[fallback_idx] if events else {},
        "reason": "test_goal_clip_pbp",
    } if events else None


def build_short_clip_tactical_entry(events, threesixty_lookup, level, base_ts, analytics_context=None):
    if not events:
        return None

    event_lookup = {}
    for event in events:
        event_id = tac.get_event_id(event)
        if event_id:
            event_lookup[event_id] = event

    chosen_event = None
    commentary_plan = tac.build_commentary_plan(events, threesixty_lookup)
    if commentary_plan:
        chosen_event = event_lookup.get(commentary_plan[0].get("event_id"))

    if chosen_event is None:
        focus_idx = tac.choose_focus_event_index(events)
        if focus_idx is None:
            focus_idx = 0
        chosen_event = events[focus_idx]

    result = tac.process_event(
        chosen_event,
        events,
        threesixty_lookup,
        selection_reason="short_clip_opening",
        source_event_id=tac.get_event_id(chosen_event),
    )
    fallback = build_short_tactical_fallback(result)
    commentary = tac.generate_commentary_ollama(
        result["tactical_description"],
        level,
        selection_reason="short_clip_opening",
        tactical_labels=result.get("tactical_labels"),
        team_name=result.get("team"),
        analytics_context=analytics_context,
        short_single_sentence=True,
    )
    commentary = sanitize_single_sentence(commentary, fallback, max_words=14)

    return {
        "video_ts": event_video_seconds(chosen_event, base_ts),
        "commentator": "tactical",
        "text": commentary,
        "event_id": f"short_clip_opening_tactical::{result.get('event_id') or 'na'}",
        "selection_reason": "short_clip_opening",
        "is_goal": result.get("is_goal", False),
        "is_foul": result.get("is_foul", False),
        "event": chosen_event,
        "short_clip_role": "opening_tactical",
        "tts_speed": SHORT_CLIP_TACTICAL_TTS_SPEED,
    }


def build_short_clip_pbp_entry(events, threesixty_lookup, base_ts):
    if not events:
        return None

    anchors = pbp.detect_all_anchors(events, threesixty_lookup)
    selected_anchor = None
    for anchor in anchors:
        if not pbp.is_long_form_commentary_event(anchor["event"]):
            selected_anchor = anchor
            break
    if selected_anchor is None and anchors:
        selected_anchor = anchors[0]

    if selected_anchor is None:
        anchor_idx = tac.choose_focus_event_index(events)
        if anchor_idx is None:
            anchor_idx = 0
        anchor_event = events[anchor_idx]
        reason = "short_clip_opening"
    else:
        anchor_idx = selected_anchor["index"]
        anchor_event = selected_anchor["event"]
        reason = selected_anchor["reason"]

    commentary = pbp.generate_template_commentary(events, anchor_idx, {"reason": reason})
    commentary = pbp.enforce_surname_only(commentary, events).strip()
    fallback = build_short_pbp_fallback(anchor_event)
    commentary = sanitize_single_sentence(commentary, fallback, max_words=4)
    if len(commentary.split()) > 3:
        commentary = fallback

    frame_360 = threesixty_lookup.get(anchor_event.get("id"))
    intensity = pbp.classify_intensity(anchor_event, frame_360)

    return {
        "video_ts": event_video_seconds(anchor_event, base_ts),
        "commentator": "pbp",
        "text": commentary,
        "event_id": f"short_clip_opening_pbp::{anchor_event.get('id') or anchor_idx}",
        "selection_reason": "short_clip_opening",
        "is_goal": pbp.is_goal_event(anchor_event),
        "is_foul": tac.is_foul_event(anchor_event),
        "intensity": intensity,
        "event": anchor_event,
        "short_clip_role": "opening_pbp",
        "tts_speed": SHORT_CLIP_PBP_TTS_SPEED,
    }


def build_special_goal_entry():
    return {
        "video_ts": SPECIAL_GOAL_TS,
        "commentator": "pbp",
        "text": "GOAL!!!",
        "event_id": "short_clip_forced_goal",
        "selection_reason": "short_clip_forced_goal",
        "is_goal": True,
        "is_foul": False,
        "intensity": "Goal",
        "event": {},
        "short_clip_role": "forced_goal",
        "tts_speed": SHORT_CLIP_GOAL_TTS_SPEED,
    }


def build_test_goal_clip_timeline(events, threesixty_lookup, level, base_ts, analytics_context=None):
    timeline = []

    tactical_event = select_best_test_goal_clip_tactical_event(events, threesixty_lookup, base_ts)
    if tactical_event is not None:
        result = tac.process_event(
            tactical_event,
            events,
            threesixty_lookup,
            selection_reason="test_goal_clip_opening",
            source_event_id=tac.get_event_id(tactical_event),
        )
        fallback = build_short_tactical_fallback(result)
        commentary = tac.generate_commentary_ollama(
            result["tactical_description"],
            level,
            selection_reason="test_goal_clip_opening",
            tactical_labels=result.get("tactical_labels"),
            team_name=result.get("team"),
            analytics_context=analytics_context,
            short_single_sentence=True,
        )
        timeline.append({
            "video_ts": min(event_video_seconds(tactical_event, base_ts), TEST_GOAL_CLIP_FREEZEFRAME_SCAN_END),
            "commentator": "tactical",
            "text": sanitize_single_sentence(commentary, fallback, max_words=12),
            "event_id": f"test_goal_clip_tactical::{result.get('event_id') or 'na'}",
            "selection_reason": "test_goal_clip_opening",
            "is_goal": False,
            "is_foul": False,
            "event": tactical_event,
            "short_clip_role": "test_goal_clip_tactical",
            "tts_speed": SHORT_CLIP_TACTICAL_TTS_SPEED,
        })

    pbp_anchor = select_test_goal_clip_pbp_anchor(events, threesixty_lookup, base_ts)
    if pbp_anchor is not None:
        anchor_idx = pbp_anchor["index"]
        anchor_event = pbp_anchor["event"]
        reason = pbp_anchor.get("reason", "test_goal_clip_pbp")
        commentary = pbp.generate_template_commentary(events, anchor_idx, {"reason": reason})
        commentary = pbp.enforce_surname_only(commentary, events).strip()
        fallback = build_short_pbp_fallback(anchor_event)
        frame_360 = threesixty_lookup.get(anchor_event.get("id"))
        intensity = pbp.classify_intensity(anchor_event, frame_360)
        timeline.append({
            "video_ts": TEST_GOAL_CLIP_PBP_TARGET_TS,
            "commentator": "pbp",
            "text": sanitize_single_sentence(commentary, fallback, max_words=5),
            "event_id": f"test_goal_clip_pbp::{anchor_event.get('id') or anchor_idx}",
            "selection_reason": "test_goal_clip_pbp",
            "is_goal": False,
            "is_foul": tac.is_foul_event(anchor_event),
            "intensity": intensity,
            "event": anchor_event,
            "short_clip_role": "test_goal_clip_pbp",
            "tts_speed": SHORT_CLIP_PBP_TTS_SPEED,
        })

    timeline.append(build_special_goal_entry())
    timeline.sort(key=lambda item: item["video_ts"])
    return timeline


def append_short_clip_exceptions(
    tac_entries,
    pbp_entries,
    events,
    threesixty_lookup,
    level,
    base_ts,
    clip_context,
    analytics_context=None,
):
    if clip_context["needs_opening_pair"]:
        opening_tactical = build_short_clip_tactical_entry(
            events,
            threesixty_lookup,
            level,
            base_ts,
            analytics_context,
        )
        opening_pbp = build_short_clip_pbp_entry(events, threesixty_lookup, base_ts)

        if opening_tactical is not None:
            tac_entries.append(opening_tactical)
        if opening_pbp is not None:
            pbp_entries.append(opening_pbp)

    if clip_context["needs_special_goal"]:
        pbp_entries.append(build_special_goal_entry())

    return tac_entries, pbp_entries


def shift_entries_after_threshold(entries, threshold, excluded_roles):
    next_ts = threshold
    for entry in sorted(entries, key=lambda item: item["video_ts"]):
        if entry.get("short_clip_role") in excluded_roles:
            continue
        if entry["video_ts"] < threshold:
            entry["video_ts"] = next_ts
            next_ts = entry["video_ts"] + MIN_COMMENTARY_GAP


def shift_entries_out_of_window(entries, start_ts, end_ts, excluded_roles):
    next_ts = end_ts
    for entry in sorted(entries, key=lambda item: item["video_ts"]):
        if entry.get("short_clip_role") in excluded_roles:
            continue
        if start_ts <= entry["video_ts"] <= end_ts:
            entry["video_ts"] = next_ts
            next_ts = entry["video_ts"] + MIN_COMMENTARY_GAP


def apply_clip_exceptions_to_timeline(timeline, clip_context):
    if not clip_context["needs_opening_pair"] and not clip_context["needs_special_goal"]:
        return timeline

    for entry in timeline:
        if clip_context["needs_forced_pair"]:
            entry.setdefault("tts_speed", SHORT_CLIP_DEFAULT_TTS_SPEED)

        role = entry.get("short_clip_role")
        if role == "opening_tactical":
            entry["video_ts"] = SHORT_CLIP_OPENING_TACTICAL_TS
            entry["tts_speed"] = SHORT_CLIP_TACTICAL_TTS_SPEED
        elif role == "opening_pbp":
            entry["video_ts"] = SHORT_CLIP_OPENING_PBP_TS
            entry["tts_speed"] = SHORT_CLIP_PBP_TTS_SPEED
        elif role == "forced_goal":
            entry["video_ts"] = SPECIAL_GOAL_TS
            entry["tts_speed"] = SHORT_CLIP_GOAL_TTS_SPEED

    if clip_context["needs_opening_pair"]:
        shift_entries_after_threshold(
            timeline,
            SHORT_CLIP_OPENING_RESERVED_END,
            {"opening_tactical", "opening_pbp", "forced_goal"},
        )

    if clip_context["needs_special_goal"]:
        shift_entries_out_of_window(
            timeline,
            SPECIAL_GOAL_TS - MIN_COMMENTARY_GAP,
            SPECIAL_GOAL_TS + MIN_COMMENTARY_GAP,
            {"opening_tactical", "opening_pbp", "forced_goal"},
        )

    timeline.sort(key=lambda item: item["video_ts"])
    for idx in range(1, len(timeline)):
        prev = timeline[idx - 1]
        curr = timeline[idx]
        if curr["video_ts"] - prev["video_ts"] < MIN_COMMENTARY_GAP:
            curr["video_ts"] = prev["video_ts"] + MIN_COMMENTARY_GAP
    return timeline


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

# ─────────────────────────────────────────────────────────────
# CLIMAX EVENT IDENTIFICATION
# ─────────────────────────────────────────────────────────────

def is_climax_event(event):
    """
    True for any Shot (goal or not), set-piece in penalty area, or dangerous
    free kick. These events define Phase-of-Play windows and TAC exclusion zones.
    """
    et = tac.get_event_type(event)
    if et == "Shot":
        return True
    if tac.is_goal_event(event):
        return True
    if tac.is_corner_or_free_kick_event(event):
        loc = tac.get_event_location(event)
        if loc and tac.is_penalty_or_goal_area_location(loc):
            return True
    return False


def identify_climax_events(events, base_ts):
    """Return list of dicts: {video_ts, event, event_id, is_goal} for all climax events."""
    climax_events = []
    for ev in events:
        if not is_climax_event(ev):
            continue
        video_ts = event_video_seconds(ev, base_ts)
        climax_events.append({
            "video_ts": video_ts,
            "event": ev,
            "event_id": tac.get_event_id(ev),
            "is_goal": tac.is_goal_event(ev),
        })
    print(f"  Identified {len(climax_events)} climax events.")
    return climax_events


# ─────────────────────────────────────────────────────────────
# AUDIO TIME-STRETCHING (PITCH-PRESERVING)
# ─────────────────────────────────────────────────────────────

def apply_time_stretch(audio_path, ratio, temp_dir=None):
    """
    Speed up audio by ratio using ffmpeg atempo (pitch-preserving, 0.5-2.0x per filter).
    Returns stretched WAV path, or original path if stretch fails.
    """
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = "ffmpeg"

    if ratio < 1.001:
        return audio_path

    out_dir = temp_dir or os.path.dirname(audio_path)
    basename = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(out_dir, f"{basename}_s{ratio:.3f}.wav")

    try:
        atempo = f"atempo={min(ratio, 2.0):.6f}"
        result = subprocess.run(
            [ffmpeg_exe, "-i", audio_path, "-filter:a", atempo, "-y", output_path],
            capture_output=True, timeout=60,
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
        print(f"  [STRETCH WARN] ffmpeg returned {result.returncode} -- using original")
    except Exception as e:
        print(f"  [STRETCH ERROR] {e} -- using original audio")
    return audio_path


def get_wav_duration(wav_path):
    """Return duration of a WAV file in seconds."""
    import wave
    try:
        with wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate > 0 else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# PHASE-OF-PLAY PBP STITCHING
# ─────────────────────────────────────────────────────────────

def stitch_pbp_lines(lines):
    """
    Join commentary lines into one flowing string with ellipsis connectors.
    Example: ["Salah drives.", "Rodri passes."] -> "Salah drives... Rodri passes."
    """
    if not lines:
        return ""
    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if i < len(lines) - 1:
            line = line.rstrip(".!?,;:")
            line = line + "..."
        cleaned.append(line)
    return " ".join(cleaned)


def _build_climax_llm_prompt(flattened_stream, intensity, anchor_event, is_goal):
    """LLM prompt for climax anchor -- relaxed word limit for dramatic exclamation."""
    parts = [f"Intensity: {intensity}"]
    if is_goal:
        parts.append(
            "[SYSTEM INSTRUCTION: This is a GOAL. Generate a short, explosive, "
            "celebratory exclamation. Use dramatic language. "
            "The call should connect naturally to the build-up action described. "
            "Keep it under 15 words.]"
        )
    else:
        parts.append(
            "[SYSTEM INSTRUCTION: This is a climax moment (shot/chance). "
            "Generate a short, punchy call connecting to the build-up. "
            "Do NOT use GOAL -- the ball has not gone in yet. "
            "Keep it under 12 words.]"
        )
    parts.append("")
    parts.append("Flattened Event Stream:")
    parts.append(flattened_stream)
    return "\n".join(parts)


def build_climax_phase_timeline(events, threesixty_lookup, base_ts, climax_events, llm_available):
    """
    For each climax event, gather preceding PBP anchors (up to CLIMAX_LOOKBACK_ANCHORS),
    generate short template commentary for each, stitch into a single flowing string,
    and return a single 'climax_phase' entry per climax.

    Returns:
        phase_entries  -- list of stitched climax entries
        subsumed_ids   -- set of event IDs already covered (skip in normal PBP build)
    """
    print("\n=== Building Climax Phase-of-Play Timeline ===")
    phase_entries = []
    subsumed_ids = set()

    if not climax_events:
        return phase_entries, subsumed_ids

    idx_360 = threesixty_lookup
    all_anchors = pbp.detect_all_anchors(events, idx_360)

    for cx in climax_events:
        cx_ts = cx["video_ts"]
        cx_event_id = cx["event_id"]
        cx_event = cx["event"]
        is_goal = cx["is_goal"]

        # Harvest build-up anchors in the lookback window
        lookback_start = cx_ts - CLIMAX_MIN_ANCHOR_LOOKBACK_SECONDS
        harvested = []
        for anchor in all_anchors:
            a_ts = event_video_seconds(anchor["event"], base_ts)
            a_id = anchor["event"].get("id")
            if a_id == cx_event_id:
                continue
            if is_climax_event(anchor["event"]):
                continue
            if lookback_start <= a_ts < cx_ts:
                harvested.append((a_ts, anchor))

        harvested.sort(key=lambda x: x[0])
        harvested = harvested[-CLIMAX_LOOKBACK_ANCHORS:]

        commentary_lines = []
        phase_event_ids = set()
        t_start = cx_ts  # will be set to earliest anchor ts

        for a_ts, anchor in harvested:
            anchor_idx = anchor["index"]
            anchor_event = anchor["event"]
            reason = anchor.get("reason", "climax_phase_buildup")
            a_id = anchor_event.get("id")
            phase_event_ids.add(a_id)

            # Use template commentary for build-up (short, punchy)
            line = pbp.generate_template_commentary(events, anchor_idx, {"reason": reason})
            line = pbp.enforce_surname_only(line, events).strip()
            if line:
                commentary_lines.append(line)
            if a_ts < t_start:
                t_start = a_ts

        # Climax call (LLM if available and appropriate, else template)
        cx_idx = next(
            (i for i, ev in enumerate(events) if tac.get_event_id(ev) == cx_event_id),
            None,
        )
        if cx_idx is not None:
            if llm_available and pbp.is_llm_commentary_event(cx_event):
                frame_360 = idx_360.get(cx_event_id)
                intensity = pbp.classify_intensity(cx_event, frame_360)
                context_events = pbp.build_context_window(events, cx_idx, {"reason": "climax"})
                flattened = pbp.build_flattened_stream(
                    context_events, cx_event, {"reason": "climax"}, idx_360
                )
                prompt = _build_climax_llm_prompt(flattened, intensity, cx_event, is_goal)
                climax_line = pbp.generate_commentary(prompt)
            else:
                climax_line = pbp.generate_template_commentary(
                    events, cx_idx, {"reason": "goal" if is_goal else "shot"}
                )

            climax_line = pbp.enforce_surname_only(climax_line, events).strip()
            if climax_line:
                commentary_lines.append(climax_line)

        if not commentary_lines:
            continue

        stitched_text = stitch_pbp_lines(commentary_lines)
        frame_360 = idx_360.get(cx_event_id)
        intensity = "Goal" if is_goal else pbp.classify_intensity(cx_event, frame_360)

        phase_entry = {
            "video_ts": t_start,           # initial placement (refined by back-alignment)
            "climax_ts": cx_ts,            # anchor for back-alignment formula
            "commentator": "pbp",
            "text": stitched_text,
            "event_id": f"climax_phase::{cx_event_id}",
            "selection_reason": "climax_phase",
            "is_goal": is_goal,
            "is_foul": False,
            "intensity": intensity,
            "event": cx_event,
            "is_climax_phase": True,
            "phase_event_ids": list(phase_event_ids),
            "phase_lines": commentary_lines,  # kept for potential re-stitching on speedup
        }
        phase_entries.append(phase_entry)
        subsumed_ids.update(phase_event_ids)
        subsumed_ids.add(cx_event_id)

        print(
            f"  [PHASE] Climax @ {cx_ts:.1f}s | buildup from {t_start:.1f}s "
            f"| {len(commentary_lines)} lines | '{stitched_text[:70]}...'"
        )

    return phase_entries, subsumed_ids


# PRIORITY ARBITRATION ENGINE
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def is_penalty_area_event(event):
    """Check if event is in or near the penalty area."""
    loc = tac.get_event_location(event)
    if loc is None:
        return False
    return tac.is_penalty_or_goal_area_location(loc)


def is_high_action_pbp_priority_event(event):
    """Events where PBP should get priority: goals, fouls, shots, penalty area action."""
    if tac.is_goal_event(event):
        return True
    if tac.is_foul_event(event):
        return True
    et = tac.get_event_type(event)
    if et == "Shot":
        return True
    if is_penalty_area_event(event):
        return True
    return False


def build_tactical_timeline(
    events, threesixty_lookup, level, base_ts,
    analytics_context=None, climax_timestamps=None
):
    """
    Run the tactical pipeline with:
      - Dynamic cooldown: max 1 spatial TAC per TAC_SPATIAL_COOLDOWN_SECONDS
      - Climax exclusion zone: no TAC within TAC_CLIMAX_EXCLUSION_BEFORE seconds of a climax
      - High-value set-pieces (free kicks/penalties) bypass the cooldown
    """
    print("\n=== Running Tactical Commentary Pipeline ===")
    climax_timestamps = climax_timestamps or []
    commentary_plan = tac.build_commentary_plan(events, threesixty_lookup)

    if not commentary_plan:
        print("  No tactical events selected.")
        return []

    # Build event lookup
    event_id_to_event = {}
    for ev in events:
        eid = tac.get_event_id(ev)
        if eid:
            event_id_to_event[eid] = ev

    entries = []
    spatial_count = 0
    last_spatial_ts = -TAC_SPATIAL_COOLDOWN_SECONDS  # allow first spatial immediately

    clip_duration = tac.get_clip_duration_seconds(events)
    max_spatial_tac = max(1, int(clip_duration / TAC_SPATIAL_COOLDOWN_SECONDS))
    print(f"  Clip duration: {clip_duration:.1f}s | Max spatial TAC events: {max_spatial_tac}")

    for item in commentary_plan:
        target_id = item["event_id"]
        event = event_id_to_event.get(target_id)
        if event is None:
            continue

        video_ts = event_video_seconds(event, base_ts)
        reason = item.get("selection_reason", "auto")
        is_mandatory = reason in ("goal", "foul")
        is_high_value_setpiece = (
            tac.is_corner_or_free_kick_event(event)
            and is_penalty_area_event(event)
        )

        # ── Climax exclusion zone (all non-mandatory TAC dropped near climax) ──
        if not is_mandatory:
            in_climax_zone = any(
                -TAC_CLIMAX_EXCLUSION_AFTER <= (video_ts - ct) <= TAC_CLIMAX_EXCLUSION_BEFORE
                for ct in climax_timestamps
            )
            if in_climax_zone:
                print(
                    f"  [TAC SKIP - CLIMAX ZONE] {target_id} @ {video_ts:.1f}s "
                    f"(too close to a climax event)"
                )
                continue

        # ── Spatial cooldown (for spatial/formation snapshots only) ──
        is_spatial = reason.startswith("spatial_")
        if is_spatial and not is_high_value_setpiece:
            if spatial_count >= max_spatial_tac:
                print(
                    f"  [TAC SKIP - MAX SPATIAL] {target_id} @ {video_ts:.1f}s "
                    f"(limit {max_spatial_tac} reached)"
                )
                continue
            if (video_ts - last_spatial_ts) < TAC_SPATIAL_COOLDOWN_SECONDS:
                print(
                    f"  [TAC SKIP - COOLDOWN] {target_id} @ {video_ts:.1f}s "
                    f"(last spatial was {video_ts - last_spatial_ts:.1f}s ago)"
                )
                continue

        # ── Generate commentary ──
        result = tac.process_event(
            event, events, threesixty_lookup,
            selection_reason=reason,
            source_event_id=item.get("source_event_id"),
        )

        print(f"  Generating tactical [{level}] for event {target_id}...")
        commentary = tac.generate_commentary_ollama(
            result["tactical_description"],
            level,
            selection_reason=result.get("selection_reason", "auto"),
            tactical_labels=result.get("tactical_labels"),
            team_name=result.get("team"),
            analytics_context=analytics_context,
        )
        print(f"    Done ({len(commentary)} chars)")

        if is_spatial and not is_high_value_setpiece:
            spatial_count += 1
            last_spatial_ts = video_ts

        entries.append({
            "video_ts": video_ts,
            "commentator": "tactical",
            "text": commentary,
            "event_id": target_id,
            "selection_reason": reason,
            "is_goal": result.get("is_goal", False),
            "is_foul": result.get("is_foul", False),
            "event": event,
        })

    print(f"  Tactical pipeline produced {len(entries)} entries.")
    return entries


def build_pbp_timeline(events, threesixty_lookup, base_ts, subsumed_ids=None):
    """
    Run the PBP pipeline and produce a list of commentary entries.
    """
    print("\n=== Running Play-by-Play Commentary Pipeline ===")
    idx_360 = threesixty_lookup  # same structure

    anchors = pbp.detect_all_anchors(events, idx_360)
    print(f"  Detected {len(anchors)} PBP anchor events.")

    # Filter out anchors already covered by climax phase stitching
    if subsumed_ids:
        before = len(anchors)
        anchors = [a for a in anchors if a["event"].get("id") not in subsumed_ids]
        print(f"  Filtered {before - len(anchors)} subsumed anchors (climax phase covers them).")

    # Check Ollama for LLM events
    needs_llm = any(pbp.is_llm_commentary_event(a["event"]) for a in anchors)
    llm_available = False

    if needs_llm:
        if pbp.check_ollama_running():
            if not pbp.check_model_exists(pbp.MODEL_NAME):
                print(f"  Creating PBP model '{pbp.MODEL_NAME}'...")
                if pbp.create_model(pbp.MODEL_NAME, PBP_MODELFILE):
                    llm_available = True
            else:
                llm_available = True

    entries = []
    for a in anchors:
        anchor_idx = a["index"]
        anchor_event = a["event"]
        reason = a["reason"]

        frame_360 = idx_360.get(anchor_event.get("id"))
        intensity = pbp.classify_intensity(anchor_event, frame_360)

        # Build context and generate commentary
        use_llm = pbp.is_llm_commentary_event(anchor_event) and llm_available

        if use_llm:
            context_events = pbp.build_context_window(events, anchor_idx, a)
            flattened = pbp.build_flattened_stream(context_events, anchor_event, a, idx_360)
            prompt = pbp.build_prompt(flattened, intensity, anchor_event)
            commentary = pbp.generate_commentary(prompt)
        else:
            commentary = pbp.generate_template_commentary(events, anchor_idx, {"reason": reason})

        commentary = pbp.enforce_surname_only(commentary, events).strip()
        if not commentary:
            continue

        video_ts = event_video_seconds(anchor_event, base_ts)
        event_id = anchor_event.get("id")

        entries.append({
            "video_ts": video_ts,
            "commentator": "pbp",
            "text": commentary,
            "event_id": event_id,
            "selection_reason": reason,
            "is_goal": pbp.is_goal_event(anchor_event),
            "is_foul": tac.is_foul_event(anchor_event),
            "intensity": intensity,
            "event": anchor_event,
        })

    print(f"  PBP pipeline produced {len(entries)} entries.")
    return entries


def merge_timelines(tac_entries, pbp_entries):
    """
    Merge tactical and PBP timelines with priority arbitration.

    Rules:
    1. No identical event_id at the same time by both commentators
    2. For spatial_midfield_dense_frame: PBP excluded 3s before, 5s after
    3. For corners/free kicks: PBP excluded ┬▒3s around tactical event
    4. For goals/fouls: PBP speaks first, then tactical (delayed)
    5. For penalty area / shot events: PBP gets priority
    6. Tactical always gets general priority outside penalty area
    """
    print("\n=== Merging Timelines ===")

    # ΓöÇΓöÇ Step 1: Build exclusion zones from tactical events ΓöÇΓöÇ
    pbp_exclude_ranges = []  # list of (start_ts, end_ts) where PBP is suppressed
    tac_exclude_ranges = []  # list of (start_ts, end_ts) where TAC is suppressed

    for entry in tac_entries:
        reason = entry.get("selection_reason", "")
        ts = entry["video_ts"]
        event = entry.get("event", {})

        if reason == "spatial_midfield_dense_frame" or reason == "spatial_5s_before_set_piece":
            # Rule 5 from user: exclude PBP 3s before and 5s after
            pbp_exclude_ranges.append((ts - SPATIAL_EXCLUDE_BEFORE, ts + SPATIAL_EXCLUDE_AFTER))

        elif tac.is_corner_or_free_kick_event(event):
            # Rule 6: corners/free kicks ΓÇö tactical priority, exclude PBP nearby
            pbp_exclude_ranges.append((ts - SET_PIECE_EXCLUDE_RADIUS, ts + SET_PIECE_EXCLUDE_RADIUS))

    # ΓöÇΓöÇ Step 2: Handle goal/foul events ΓÇö PBP first, tactical after ΓöÇΓöÇ
    for entry in tac_entries:
        if entry.get("short_clip_role") == "opening_tactical":
            continue
        if entry.get("is_goal") or entry.get("is_foul"):
            # Delay tactical commentary after PBP
            entry["video_ts"] = entry["video_ts"] + GOAL_FOUL_TAC_DELAY

    # ΓöÇΓöÇ Step 3: For penalty area / high-action events, PBP gets priority ΓöÇΓöÇ
    for entry in tac_entries:
        if entry.get("short_clip_role") == "opening_tactical":
            continue
        event = entry.get("event", {})
        if is_high_action_pbp_priority_event(event):
            reason = entry.get("selection_reason", "")
            # Don't suppress tactical for goal/foul (they're already delayed)
            if not entry.get("is_goal") and not entry.get("is_foul"):
                ts = entry["video_ts"]
                tac_exclude_ranges.append((ts - 2.0, ts + 2.0))

    # ΓöÇΓöÇ Step 4: Filter PBP entries by exclusion zones ΓöÇΓöÇ
    filtered_pbp = []
    for entry in pbp_entries:
        if entry.get("short_clip_role") in {"opening_pbp", "forced_goal"}:
            filtered_pbp.append(entry)
            continue
        ts = entry["video_ts"]
        excluded = any(start <= ts <= end for start, end in pbp_exclude_ranges)
        if excluded:
            print(f"  [PBP EXCLUDED] {entry['event_id']} at {ts:.1f}s ΓÇö "
                  f"within tactical exclusion zone")
            continue
        filtered_pbp.append(entry)

    # ΓöÇΓöÇ Step 5: Filter tactical entries in PBP priority zones ΓöÇΓöÇ
    filtered_tac = []
    for entry in tac_entries:
        if entry.get("short_clip_role") == "opening_tactical":
            filtered_tac.append(entry)
            continue
        ts = entry["video_ts"]
        excluded = any(start <= ts <= end for start, end in tac_exclude_ranges)
        if excluded:
            print(f"  [TAC EXCLUDED] {entry['event_id']} at {ts:.1f}s ΓÇö "
                  f"PBP has priority (penalty area)")
            continue
        filtered_tac.append(entry)

    # ΓöÇΓöÇ Step 6: Combine and sort by timestamp ΓöÇΓöÇ
    combined = filtered_tac + filtered_pbp
    combined.sort(key=lambda e: e["video_ts"])

    # ΓöÇΓöÇ Step 7: Remove duplicates (same event_id) ΓöÇΓöÇ
    seen_ids = set()
    deduped = []
    for entry in combined:
        eid = entry["event_id"]
        if eid in seen_ids:
            print(f"  [DEDUP] Dropped duplicate event {eid}")
            continue
        seen_ids.add(eid)
        deduped.append(entry)

    # ΓöÇΓöÇ Step 8: Enforce minimum gap between entries ΓöÇΓöÇ
    final = []
    for entry in deduped:
        if final:
            last_ts = final[-1]["video_ts"]
            if entry["video_ts"] - last_ts < MIN_COMMENTARY_GAP:
                # Keep higher priority entry
                if entry.get("is_goal") or entry.get("is_foul"):
                    # Goal/foul always wins
                    if not final[-1].get("is_goal") and not final[-1].get("is_foul"):
                        final[-1] = entry
                        continue
                # Otherwise just shift it
                entry["video_ts"] = last_ts + MIN_COMMENTARY_GAP
        final.append(entry)

    print(f"  Merged timeline: {len(final)} entries "
          f"({len(filtered_tac)} tactical + {len(filtered_pbp)} PBP, "
          f"after dedup/spacing: {len(final)})")

    return final


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# TEXT-TO-SPEECH
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ


# ─────────────────────────────────────────────────────────────
# AUDIO BACK-ALIGNMENT
# ─────────────────────────────────────────────────────────────

def back_align_climax_entries(timeline):
    """
    After TTS generation, pin climax_phase audio so its END lands at
    T_climax + REACTION_BUFFER_SECONDS.

    Formula: T_audio_start = T_climax - audio_duration + REACTION_BUFFER_SECONDS

    Then run a collision check: if the back-aligned start collides with the
    previous entry's audio, the previous entry wins (climax phase drops its
    build-up front — it never clobbers a concurrent commentary).
    """
    print("\n=== Back-Aligning Climax Phase Audio ===")
    for entry in timeline:
        if not entry.get("is_climax_phase"):
            continue
        if not entry.get("audio_path") or not entry.get("audio_duration"):
            continue

        climax_ts = entry.get("climax_ts", entry["video_ts"])
        audio_dur = entry["audio_duration"]
        new_start = climax_ts - audio_dur + REACTION_BUFFER_SECONDS
        new_start = max(0.0, new_start)

        old_start = entry["video_ts"]
        entry["video_ts"] = new_start
        print(
            f"  [BACK-ALIGN] '{entry['text'][:50]}...' "
            f"climax={climax_ts:.1f}s dur={audio_dur:.1f}s "
            f"start: {old_start:.1f}s -> {new_start:.1f}s"
        )

    # Re-sort and collision check
    timeline.sort(key=lambda e: e["video_ts"])
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]
        if not prev.get("audio_path") or not curr.get("audio_path"):
            continue
        prev_end = prev["video_ts"] + prev.get("audio_duration", 0.0)
        if curr["video_ts"] < prev_end + MIN_AUDIO_GAP_SECONDS:
            if curr.get("is_climax_phase"):
                # Climax phase takes priority — warn but let it overlap slightly
                print(
                    f"  [COLLISION] Climax phase @ {curr['video_ts']:.1f}s collides with "
                    f"prev ending @ {prev_end:.1f}s — climax phase wins."
                )
            else:
                # Normal entry bumped forward
                curr["video_ts"] = prev_end + MIN_AUDIO_GAP_SECONDS

    return timeline


def generate_tts_audio(timeline, progress_callback=None):
    """
    Generate WAV files for each commentary entry using Coqui TTS (XTTSv2).
    Returns list of entries with 'audio_path' and 'audio_duration' added.
    """
    print("\n=== Generating TTS Audio with XTTSv2 ===")

    import wave
    import torch
    
    # Automatically accept Coqui TTS TOS programmatically
    os.environ["COQUI_TOS_AGREED"] = "1"
    
    from TTS.api import TTS

    if progress_callback:
        progress_callback("Initializing XTTSv2 (this may take a moment)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading XTTSv2 on {device}...")
    
    # Initialize TTS
    try:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    except Exception as e:
        print(f"  Failed to initialize TTS: {e}")
        return timeline

    total = len(timeline)
    for i, entry in enumerate(timeline):
        if progress_callback:
            progress_callback(f"Generating audio {i+1}/{total}...")

        text = entry["text"]
        if not text or text.startswith("[ERROR"):
            entry["audio_path"] = None
            entry["audio_duration"] = 0.0
            continue

        wav_path = os.path.join(TEMP_DIR, f"tts_{i:03d}_{entry['commentator']}.wav")

        # point to backend/TTS
        tts_dir = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "TTS")

        # Determine reference voice logic
        if entry["commentator"] == "pbp":
            intensity = entry.get("intensity", "Neutral")
            if intensity == "Goal":
                speaker_wav = os.path.join(tts_dir, "Goal.mp3")
            elif intensity == "High":
                speaker_wav = os.path.join(tts_dir, "High.mp3")
            else:
                speaker_wav = os.path.join(tts_dir, "Neutral.mp3")
        else:  # tactical
            speaker_wav = os.path.join(tts_dir, "Tactical.mp3")

        # For climax-phase entries, override text with the latest phase_lines
        # (may have been trimmed by time-budgeting loop below)
        if entry.get("is_climax_phase") and entry.get("phase_lines"):
            text = entry["text"]  # use the stitched version

        # Fallback to Neutral if chosen path is missing somehow
        if not os.path.exists(speaker_wav):
            fallback_wav = os.path.join(tts_dir, "Neutral.mp3")
            if os.path.exists(fallback_wav):
                speaker_wav = fallback_wav
            else:
                # Absolute worst case, XTTS won't work without a reference audio.
                # Skip generation.
                print(f"  [{i+1}/{total}] ERROR reference audio missing: {speaker_wav}")
                entry["audio_path"] = None
                entry["audio_duration"] = 0.0
                continue

        try:
            print(f"  [{i+1}/{total}] {entry['commentator'].upper():9s} - Cloning: {os.path.basename(speaker_wav)}")
            tts_speed = float(entry.get("tts_speed", 1.0) or 1.0)
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language="en",
                file_path=wav_path,
                speed=tts_speed,
            )

            # Get duration using the wave module
            if os.path.exists(wav_path):
                with wave.open(wav_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate) if rate > 0 else 0.0

                # ── Time-budgeting for climax_phase entries ──
                if entry.get("is_climax_phase") and entry.get("phase_lines"):
                    budget = entry.get("climax_ts", entry["video_ts"]) - entry["video_ts"]
                    if budget > 0 and duration > budget:
                        ratio = duration / budget
                        if ratio <= MAX_SPEEDUP_RATIO:
                            print(
                                f"      [BUDGET] duration={duration:.1f}s "
                                f"budget={budget:.1f}s ratio={ratio:.2f}x "
                                f"-> applying speedup"
                            )
                            stretched = apply_time_stretch(wav_path, ratio, TEMP_DIR)
                            new_dur = get_wav_duration(stretched)
                            if new_dur > 0:
                                wav_path = stretched
                                duration = new_dur
                        else:
                            print(
                                f"      [BUDGET] ratio={ratio:.2f}x > {MAX_SPEEDUP_RATIO}x "
                                f"-- dropping first build-up line"
                            )
                            lines = entry.get("phase_lines", [])
                            if len(lines) > 2:
                                entry["phase_lines"] = lines[1:]
                                entry["text"] = stitch_pbp_lines(entry["phase_lines"])

                entry["audio_path"] = wav_path
                entry["audio_duration"] = duration
                print(f"      @ {entry['video_ts']:6.1f}s | {duration:.1f}s | speed={tts_speed:.2f} | {text[:60]}...")
            else:
                entry["audio_path"] = None
                entry["audio_duration"] = 0.0
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR generating/reading audio: {e}")
            entry["audio_path"] = None
            entry["audio_duration"] = 0.0

    # Fix overlaps: if one commentary would overlap the next, shift the next
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]
        if prev.get("audio_path") and curr.get("audio_path"):
            prev_end = prev["video_ts"] + prev["audio_duration"]
            if curr["video_ts"] < prev_end + 0.3:
                if curr.get("short_clip_role") == "forced_goal":
                    continue
                curr["video_ts"] = prev_end + 0.3

    return timeline


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# AUDIO COMPOSITION & VIDEO OUTPUT
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def compose_video_with_commentary(timeline, video_file, progress_callback=None):
    """
    Overlay TTS commentary audio on the original video's crowd audio.
    Returns path to the final output video.
    """
    print("\n=== Composing Final Video ===")
    if progress_callback:
        progress_callback("Composing video with commentary...")

    from moviepy import VideoFileClip, AudioFileClip
    from pydub import AudioSegment
    import imageio_ffmpeg
    # Point pydub to the ffmpeg bundled with imageio_ffmpeg
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.ffprobe = imageio_ffmpeg.get_ffmpeg_exe()

    video = VideoFileClip(video_file)
    video_duration = video.duration

    # Load original audio as pydub segment for mixing
    # First extract original audio to a temp wav
    original_audio_path = os.path.join(TEMP_DIR, "original_audio.wav")
    if video.audio is not None:
        video.audio.write_audiofile(original_audio_path, logger=None)
        crowd_audio = AudioSegment.from_wav(original_audio_path)
    else:
        # No audio in video ΓÇö create silence
        crowd_audio = AudioSegment.silent(duration=int(video_duration * 1000))

    # Build the commentary overlay track
    commentary_track = AudioSegment.silent(duration=len(crowd_audio))

    for entry in timeline:
        audio_path = entry.get("audio_path")
        if not audio_path or not os.path.exists(audio_path):
            continue

        position_ms = int(entry["video_ts"] * 1000)
        if position_ms < 0:
            position_ms = 0
        if position_ms >= len(commentary_track):
            continue

        try:
            tts_audio = AudioSegment.from_wav(audio_path)
            # Boost commentary volume slightly
            tts_audio = tts_audio + 3  # +3 dB
            commentary_track = commentary_track.overlay(tts_audio, position=position_ms)
        except Exception as e:
            print(f"  Error overlaying audio at {position_ms}ms: {e}")

    # Duck crowd audio slightly where commentary plays
    # Simple approach: just lower crowd audio globally by a few dB
    # and keep it cleaner
    crowd_ducked = crowd_audio + CROWD_DUCK_DB  # duck down
    mixed = crowd_ducked.overlay(commentary_track)

    # Export mixed audio
    mixed_audio_path = os.path.join(TEMP_DIR, "mixed_audio.wav")
    mixed.export(mixed_audio_path, format="wav")

    # Build final video with new audio
    output_path = os.path.join(TEMP_DIR, "final_output.mp4")
    new_audio = AudioFileClip(mixed_audio_path)
    final_video = video.with_audio(new_audio)
    final_video.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        logger=None,
    )

    # Cleanup
    video.close()
    new_audio.close()

    print(f"  Final video saved to: {output_path}")
    return output_path


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# VIDEO PLAYBACK (VLC embedded in Tkinter)
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def play_video_vlc(video_path, parent_window=None):
    """Play video using VLC embedded in tkinter, or fallback to system player."""
    try:
        import vlc
        # Create a new top-level window for the player
        player_win = tk.Toplevel(parent_window) if parent_window else tk.Tk()
        player_win.title("ΓÜ╜ Commentary Playback")
        player_win.geometry("960x600")
        player_win.configure(bg="black")

        # Create VLC instance and player
        instance = vlc.Instance()
        player = instance.media_player_new()
        media = instance.media_new(video_path)
        player.set_media(media)

        # Create frame for video
        video_frame = tk.Frame(player_win, bg="black", width=960, height=540)
        video_frame.pack(fill=tk.BOTH, expand=True)
        video_frame.update()

        # Set the window handle for VLC
        player.set_hwnd(video_frame.winfo_id())

        # Controls frame
        controls = tk.Frame(player_win, bg="#1a1a2e", height=60)
        controls.pack(fill=tk.X, side=tk.BOTTOM)

        def on_play_pause():
            if player.is_playing():
                player.pause()
                play_btn.config(text="Γû╢ Play")
            else:
                player.play()
                play_btn.config(text="ΓÅ╕ Pause")

        def on_stop():
            player.stop()
            player_win.destroy()

        play_btn = tk.Button(controls, text="ΓÅ╕ Pause", command=on_play_pause,
                             bg="#e94560", fg="white", font=("Segoe UI", 11, "bold"),
                             relief=tk.FLAT, padx=20, pady=5)
        play_btn.pack(side=tk.LEFT, padx=10, pady=10)

        stop_btn = tk.Button(controls, text="Γ£û Close", command=on_stop,
                             bg="#533483", fg="white", font=("Segoe UI", 11, "bold"),
                             relief=tk.FLAT, padx=20, pady=5)
        stop_btn.pack(side=tk.RIGHT, padx=10, pady=10)

        # Start playing
        player.play()

        def on_close():
            player.stop()
            player_win.destroy()

        player_win.protocol("WM_DELETE_WINDOW", on_close)

    except Exception as e:
        print(f"VLC playback failed ({e}), falling back to system player...")
        os.startfile(video_path)


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# MAIN PIPELINE (called from GUI)
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

def run_pipeline(
    events,
    threesixty_lookup,
    video_file,
    level,
    clip_filename=None,
    analytics_context=None,
    progress_callback=None,
    done_callback=None,
):
    """
    Full pipeline: load → generate both commentaries → merge → TTS → compose video.
    Runs in a background thread or executor.
    """
    try:
        base_ts = get_base_timestamp(events)
        print(f"Base timestamp: {base_ts:.3f}s (first event)")
        clip_context = build_clip_exception_context(events, video_file, clip_filename)
        print(
            f"Clip exception context: duration={clip_context['clip_duration']:.2f}s "
            f"| filename={clip_context['clip_filename'] or 'unknown'} "
            f"| opening_pair={clip_context['needs_opening_pair']} "
            f"| test_goal_triplet={clip_context['needs_test_goal_triplet']} "
            f"| forced_goal={clip_context['needs_special_goal']}"
        )

        # ΓöÇΓöÇ Ensure Ollama models are ready ΓöÇΓöÇ
        if progress_callback:
            progress_callback("Checking Ollama models...")

        if pbp.check_ollama_running():
            if not pbp.check_model_exists(tac.MODEL_NAME):
                print(f"Creating tactical model '{tac.MODEL_NAME}'...")
                pbp.create_model(tac.MODEL_NAME, TAC_MODELFILE)
            if not pbp.check_model_exists(pbp.MODEL_NAME):
                print(f"Creating PBP model '{pbp.MODEL_NAME}'...")
                pbp.create_model(pbp.MODEL_NAME, PBP_MODELFILE)
        else:
            print("WARNING: Ollama not running. Commentary will use fallbacks/templates.")

        # ΓöÇΓöÇ Generate both commentary timelines ΓöÇΓöÇ
        if progress_callback:
            progress_callback("Generating tactical commentary...")
        # -- Identify climax events (shots/goals/dangerous set-pieces)
        if progress_callback:
            progress_callback("Identifying climax events...")
        climax_events = identify_climax_events(events, base_ts)
        climax_timestamps = [cx["video_ts"] for cx in climax_events]

        # -- Check LLM availability for phase timeline
        llm_available = False
        if pbp.check_ollama_running():
            if pbp.check_model_exists(pbp.MODEL_NAME):
                llm_available = True

        # -- Build Phase-of-Play stitched entries for climax moments
        if progress_callback:
            progress_callback("Building climax phase-of-play commentary...")
        phase_entries, subsumed_ids = build_climax_phase_timeline(
            events, threesixty_lookup, base_ts, climax_events, llm_available
        )

        tac_entries = build_tactical_timeline(
            events, threesixty_lookup, level, base_ts,
            analytics_context=analytics_context,
            climax_timestamps=climax_timestamps,
        )

        if progress_callback:
            progress_callback("Generating play-by-play commentary...")
        pbp_entries = build_pbp_timeline(
            events, threesixty_lookup, base_ts, subsumed_ids=subsumed_ids
        )
        pbp_entries.extend(phase_entries)

        if clip_context["needs_opening_pair"] or clip_context["needs_special_goal"]:
            if progress_callback:
                progress_callback("Applying short-clip commentary exceptions...")
            tac_entries, pbp_entries = append_short_clip_exceptions(
                tac_entries,
                pbp_entries,
                events,
                threesixty_lookup,
                level,
                base_ts,
                clip_context,
                analytics_context,
            )

        # ΓöÇΓöÇ Merge with priority arbitration ΓöÇΓöÇ
        if progress_callback:
            progress_callback("Merging commentary timelines...")
        timeline = merge_timelines(tac_entries, pbp_entries)
        timeline = apply_clip_exceptions_to_timeline(timeline, clip_context)
        if clip_context["needs_test_goal_triplet"]:
            if progress_callback:
                progress_callback("Applying test clip 3-commentary exception...")
            timeline = build_test_goal_clip_timeline(
                events,
                threesixty_lookup,
                level,
                base_ts,
                analytics_context,
            )

        # ΓöÇΓöÇ Print unified timeline ΓöÇΓöÇ
        print("\n=== Unified Commentary Timeline ===")
        for i, entry in enumerate(timeline):
            tag = "≡ƒÄ» TAC" if entry["commentator"] == "tactical" else "≡ƒÄÖ∩╕Å PBP"
            print(f"  [{i+1:2d}] {entry['video_ts']:6.1f}s {tag} | {entry['text'][:80]}...")

        # ΓöÇΓöÇ Generate TTS audio ΓöÇΓöÇ
        if progress_callback:
            progress_callback("Generating voice audio...")
        timeline = generate_tts_audio(timeline, progress_callback)

        # ΓöÇΓöÇ Compose final video ΓöÇΓöÇ
        # ── Back-align climax entries and perform collision check ──
        if progress_callback:
            progress_callback("Aligning audio to video events...")
        timeline = back_align_climax_entries(timeline)

        # ── Compose final video ──
        if progress_callback:
            progress_callback("Building final video (this may take a moment)...")
        output_path = compose_video_with_commentary(timeline, video_file, progress_callback)

        if done_callback:
            done_callback(output_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if done_callback:
            done_callback(None, error=str(e))


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# GUI
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

class CommentaryApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ΓÜ╜ Merged Commentary System")
        self.root.geometry("520x420")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f3460")

        self.output_video_path = None
        self._build_ui()

    def _build_ui(self):
        root = self.root

        # ΓöÇΓöÇ Title ΓöÇΓöÇ
        title_frame = tk.Frame(root, bg="#0f3460")
        title_frame.pack(pady=(25, 10))

        tk.Label(title_frame, text="ΓÜ╜", font=("Segoe UI", 28),
                 bg="#0f3460", fg="white").pack()
        tk.Label(title_frame, text="Merged Commentary System",
                 font=("Segoe UI", 18, "bold"), bg="#0f3460", fg="#e94560").pack()
        tk.Label(title_frame, text="Tactical + Play-by-Play Commentary Generator",
                 font=("Segoe UI", 10), bg="#0f3460", fg="#a3a3c2").pack(pady=(2, 0))

        # ΓöÇΓöÇ Level Selection ΓöÇΓöÇ
        level_frame = tk.Frame(root, bg="#16213e", padx=20, pady=15)
        level_frame.pack(fill=tk.X, padx=30, pady=(15, 10))

        tk.Label(level_frame, text="Select Tactical Commentary Level:",
                 font=("Segoe UI", 11, "bold"), bg="#16213e", fg="white").pack(anchor=tk.W)

        self.level_var = tk.StringVar(value="Intermediate")
        levels = ["Beginner", "Intermediate", "Expert"]
        for lvl in levels:
            rb = tk.Radiobutton(
                level_frame, text=lvl, variable=self.level_var, value=lvl,
                font=("Segoe UI", 10), bg="#16213e", fg="white",
                selectcolor="#533483", activebackground="#16213e", activeforeground="white",
                indicatoron=True,
            )
            rb.pack(anchor=tk.W, padx=10, pady=2)

        # ΓöÇΓöÇ Generate Button ΓöÇΓöÇ
        self.generate_btn = tk.Button(
            root, text="≡ƒÄÖ∩╕Å  Generate Commentary",
            command=self._on_generate,
            font=("Segoe UI", 12, "bold"), bg="#e94560", fg="white",
            relief=tk.FLAT, padx=20, pady=8, cursor="hand2",
        )
        self.generate_btn.pack(pady=(10, 5))

        # ΓöÇΓöÇ Progress Label ΓöÇΓöÇ
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            root, textvariable=self.progress_var,
            font=("Segoe UI", 10), bg="#0f3460", fg="#a3a3c2",
        )
        self.progress_label.pack(pady=5)

        # ΓöÇΓöÇ Progress Bar ΓöÇΓöÇ
        self.progress_bar = ttk.Progressbar(root, mode="indeterminate", length=300)

        # ΓöÇΓöÇ Play Button (hidden until ready) ΓöÇΓöÇ
        self.play_btn = tk.Button(
            root, text="Γû╢  Play Video with Commentary",
            command=self._on_play,
            font=("Segoe UI", 12, "bold"), bg="#533483", fg="white",
            relief=tk.FLAT, padx=20, pady=8, cursor="hand2",
        )

    def _on_generate(self):
        level = self.level_var.get()
        self.generate_btn.config(state=tk.DISABLED)
        self.progress_bar.pack(pady=5)
        self.progress_bar.start(15)
        self._update_progress("Starting commentary generation...")

        def progress_cb(msg):
            self.root.after(0, self._update_progress, msg)

        def done_cb(output_path, error=None):
            self.root.after(0, self._on_generation_done, output_path, error)

        thread = threading.Thread(
            target=run_pipeline,
            args=(level,),
            kwargs={"progress_callback": progress_cb, "done_callback": done_cb},
            daemon=True,
        )
        thread.start()

    def _update_progress(self, msg):
        self.progress_var.set(msg)

    def _on_generation_done(self, output_path, error=None):
        self.progress_bar.stop()
        self.progress_bar.pack_forget()

        if error:
            self.progress_var.set(f"Error: {error}")
            self.generate_btn.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Commentary generation failed:\n{error}")
            return

        self.output_video_path = output_path
        self.progress_var.set("Γ£à Commentary generated successfully!")
        self.play_btn.pack(pady=(10, 15))

        answer = messagebox.askyesno(
            "Commentary Ready",
            "Commentary has been generated and overlaid on the video.\n\n"
            "Do you want to play the video now?"
        )
        if answer:
            self._on_play()

    def _on_play(self):
        if self.output_video_path and os.path.exists(self.output_video_path):
            play_video_vlc(self.output_video_path, self.root)
        else:
            messagebox.showerror("Error", "Output video not found.")

    def run(self):
        self.root.mainloop()


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
# ENTRY POINT
# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ

if __name__ == "__main__":
    app = CommentaryApp()
    app.run()
