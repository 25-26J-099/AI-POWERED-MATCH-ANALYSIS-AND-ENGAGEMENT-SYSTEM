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

TEMP_DIR = os.path.join(tempfile.gettempdir(), "merged_commentary")
os.makedirs(TEMP_DIR, exist_ok=True)


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


# ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
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


def build_tactical_timeline(events, threesixty_lookup, level, base_ts, analytics_context=None, audience_profile=None):
    """
    Run the tactical pipeline and produce a list of commentary entries.
    Each entry: {video_ts, commentator, text, event_id, selection_reason}
    """
    print("\n=== Running Tactical Commentary Pipeline ===")
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
    for item in commentary_plan:
        target_id = item["event_id"]
        event = event_id_to_event.get(target_id)
        if event is None:
            continue

        # Process the event to get tactical description
        result = tac.process_event(
            event, events, threesixty_lookup,
            selection_reason=item.get("selection_reason", "auto"),
            source_event_id=item.get("source_event_id"),
        )

        # Generate commentary at the requested level only
        print(f"  Generating tactical [{level}] for event {target_id}...")
        commentary = tac.generate_commentary_ollama(
            result["tactical_description"],
            level,
            selection_reason=result.get("selection_reason", "auto"),
            tactical_labels=result.get("tactical_labels"),
            team_name=result.get("team"),
            analytics_context=analytics_context,
            audience_profile=audience_profile,
        )
        print(f"    Done ({len(commentary)} chars)")

        video_ts = event_video_seconds(event, base_ts)

        entries.append({
            "video_ts": video_ts,
            "commentator": "tactical",
            "text": commentary,
            "event_id": target_id,
            "selection_reason": item.get("selection_reason", "auto"),
            "is_goal": result.get("is_goal", False),
            "is_foul": result.get("is_foul", False),
            "event": event,
        })

    print(f"  Tactical pipeline produced {len(entries)} entries.")
    return entries


def build_pbp_timeline(events, threesixty_lookup, base_ts):
    """
    Run the PBP pipeline and produce a list of commentary entries.
    """
    print("\n=== Running Play-by-Play Commentary Pipeline ===")
    idx_360 = threesixty_lookup  # same structure

    anchors = pbp.detect_all_anchors(events, idx_360)
    print(f"  Detected {len(anchors)} PBP anchor events.")

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
        if entry.get("is_goal") or entry.get("is_foul"):
            # Delay tactical commentary after PBP
            entry["video_ts"] = entry["video_ts"] + GOAL_FOUL_TAC_DELAY

    # ΓöÇΓöÇ Step 3: For penalty area / high-action events, PBP gets priority ΓöÇΓöÇ
    for entry in tac_entries:
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
            tts.tts_to_file(text=text, speaker_wav=speaker_wav, language="en", file_path=wav_path)

            # Get duration using the wave module
            if os.path.exists(wav_path):
                with wave.open(wav_path, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    duration = frames / float(rate) if rate > 0 else 0.0
                entry["audio_path"] = wav_path
                entry["audio_duration"] = duration
                print(f"      @ {entry['video_ts']:6.1f}s | {duration:.1f}s | {text[:60]}...")
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
    analytics_context=None,
    progress_callback=None,
    done_callback=None,
    audience_profile=None,
):
    """
    Full pipeline: load → generate both commentaries → merge → TTS → compose video.
    Runs in a background thread or executor.
    """
    try:
        base_ts = get_base_timestamp(events)
        print(f"Base timestamp: {base_ts:.3f}s (first event)")

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
        tac_entries = build_tactical_timeline(
            events,
            threesixty_lookup,
            level,
            base_ts,
            analytics_context,
            audience_profile=audience_profile,
        )

        if progress_callback:
            progress_callback("Generating play-by-play commentary...")
        pbp_entries = build_pbp_timeline(events, threesixty_lookup, base_ts)

        # ΓöÇΓöÇ Merge with priority arbitration ΓöÇΓöÇ
        if progress_callback:
            progress_callback("Merging commentary timelines...")
        timeline = merge_timelines(tac_entries, pbp_entries)

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
