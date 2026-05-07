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

# ── Import existing pipelines as modules ────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import tac_commentary as tac
import pbp_commentary as pbp

# ── Configuration ───────────────────────────────────────────
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
MAX_SILENCE_SECONDS = 6.0          # Fill real post-TTS dead air longer than this
MAX_PLANNED_SILENCE_SECONDS = MAX_SILENCE_SECONDS
GAP_FILL_TARGET_SECONDS = 5.0      # Preferred point for continuity beats inside a gap
GAP_FILL_EDGE_BUFFER_SECONDS = 2.5 # Keep fillers away from existing commentary

# -- Dynamic Tactical Cooldown
TAC_SPATIAL_COOLDOWN_SECONDS = 50.0    # Min gap between spatial TAC commentary events
TAC_CLIMAX_EXCLUSION_BEFORE = 11.0    # Block TAC N seconds before a climax event
TAC_CLIMAX_EXCLUSION_AFTER  = 2.0     # Block TAC N seconds after a climax event

# -- Climax Phase-of-Play (PBP Stitching)
CLIMAX_LOOKBACK_ANCHORS = 5           # Max anchors to stitch before a climax event
CLIMAX_MIN_BUILDUP_ANCHORS = 3        # Minimum buildup anchors required for a true phase
CLIMAX_PHASE_SEPARATOR = '... '       # Separator between stitched PBP lines
CLIMAX_MIN_ANCHOR_LOOKBACK_SECONDS = 20.0

# -- Audio Time-Budgeting & Back-Alignment
REACTION_BUFFER_SECONDS = 0.75         # Delay after climax before commentary lands
MAX_SPEEDUP_RATIO = 1.25               # Never stretch audio faster than 1.25x
MIN_AUDIO_GAP_SECONDS = 0.30           # Minimum gap between any two audio clips

TEMP_DIR = os.path.join(tempfile.gettempdir(), "merged_commentary")
os.makedirs(TEMP_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

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




# ─────────────────────────────────────────────────────────────
# CLIMAX EVENT IDENTIFICATION
# ─────────────────────────────────────────────────────────────

def is_climax_event(event):
    """True for any Shot, Goal, or dangerous set-piece in/near the penalty area."""
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
    """Return list of {video_ts, event, event_id, is_goal} for all climax events."""
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
    """Speed up audio by ratio using ffmpeg atempo (pitch-preserving)."""
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
        print(f"  [STRETCH WARN] ffmpeg {result.returncode} -- using original")
    except Exception as e:
        print(f"  [STRETCH ERROR] {e} -- using original audio")
    return audio_path


def get_wav_duration(wav_path):
    """Return WAV file duration in seconds."""
    import wave as _wave
    try:
        with _wave.open(wav_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate > 0 else 0.0
    except Exception:
        return 0.0


def audio_entry_end(entry):
    """Return the scheduled audio end time for an entry."""
    return entry.get("video_ts", 0.0) + entry.get("audio_duration", 0.0)


def is_climax_audio_entry(entry):
    return bool(entry.get("is_climax_phase") or entry.get("is_climax_direct") or entry.get("is_goal"))


def resolve_audio_collisions(timeline, min_gap=MIN_AUDIO_GAP_SECONDS):
    """
    Guarantee serialized commentary after real TTS durations are known.

    Climax audio wins over nearby non-climax audio. Routine entries are shifted
    after the previous confirmed audio, which prevents PBP/TAC double-voicing.
    """
    active = [
        entry for entry in timeline
        if entry.get("audio_path") and entry.get("audio_duration", 0.0) > 0.0
    ]
    active.sort(key=lambda e: e.get("video_ts", 0.0))

    resolved = []
    for entry in active:
        entry["drop_audio"] = False
        while resolved:
            prev = resolved[-1]
            prev_end = audio_entry_end(prev)
            if entry["video_ts"] >= prev_end + min_gap:
                break

            if is_climax_audio_entry(entry) and not is_climax_audio_entry(prev):
                print(
                    f"  [AUDIO COLLISION] Dropping {prev.get('commentator')} "
                    f"@ {prev.get('video_ts', 0.0):.1f}s so climax @ "
                    f"{entry.get('video_ts', 0.0):.1f}s is clean."
                )
                prev["drop_audio"] = True
                resolved.pop()
                continue

            if is_climax_audio_entry(prev) and not is_climax_audio_entry(entry):
                new_start = prev_end + min_gap
                print(
                    f"  [AUDIO SHIFT] Moving {entry.get('commentator')} "
                    f"{entry.get('video_ts', 0.0):.1f}s -> {new_start:.1f}s "
                    "after climax audio."
                )
                entry["video_ts"] = new_start
                break

            new_start = prev_end + min_gap
            print(
                f"  [AUDIO SHIFT] Moving {entry.get('commentator')} "
                f"{entry.get('video_ts', 0.0):.1f}s -> {new_start:.1f}s "
                "to prevent overlap."
            )
            entry["video_ts"] = new_start
            break

        if not resolved or entry["video_ts"] >= audio_entry_end(resolved[-1]) + min_gap:
            resolved.append(entry)
            continue
        resolved.append(entry)

    ordered = [entry for entry in timeline if not entry.get("drop_audio")]
    ordered.sort(key=lambda e: e.get("video_ts", 0.0))
    return ordered


# ─────────────────────────────────────────────────────────────
# PHASE-OF-PLAY PBP STITCHING
# ─────────────────────────────────────────────────────────────

def stitch_pbp_lines(lines):
    """Join commentary lines with ellipsis connectors into one flowing string."""
    if not lines:
        return ""
    cleaned = []
    for i, line in enumerate(lines):
        line = line.strip()
        if i < len(lines) - 1:
            line = line.rstrip(".!?,;:") + "..."
        cleaned.append(line)
    return " ".join(cleaned)


def _build_climax_llm_prompt(flattened_stream, intensity, anchor_event, is_goal):
    """LLM prompt for climax anchor with relaxed word limit."""
    parts = [f"Intensity: {intensity}"]
    if is_goal:
        parts.append(
            "[SYSTEM INSTRUCTION: This is a GOAL. Generate a short, explosive, "
            "celebratory exclamation that connects naturally to the build-up.]"
        )
    else:
        parts.append(
            "[SYSTEM INSTRUCTION: This is a climax moment (shot/chance). "
            "Generate a short, punchy call connecting to the build-up. "
            "Do NOT use GOAL unless the event stream explicitly says it is a goal.]"
        )
    parts.append("")
    parts.append("Flattened Event Stream:")
    parts.append(flattened_stream)
    return "\n".join(parts)


def _build_phase_buildup_prompt(flattened_stream, intensity, anchor_event):
    parts = [f"Intensity: {intensity}"]
    parts.append(
        "[SYSTEM INSTRUCTION: Generate a short, punchy play-by-play call that "
        "connects to the previous action. Use one natural broadcast sentence. "
        "Do not invent players, outcomes, goals, fouls, or assists beyond the event stream.]"
    )
    parts.append("")
    parts.append("Flattened Event Stream:")
    parts.append(flattened_stream)
    return "\n".join(parts)


def is_phase_buildup_event(event):
    return tac.get_event_type(event) in {"Pass", "Carry", "Dribble"}


def _generate_phase_anchor_line(events, idx_360, anchor_idx, anchor_event, reason, llm_available):
    if llm_available:
        try:
            frame_360 = idx_360.get(anchor_event.get("id"))
            intensity = pbp.classify_intensity(anchor_event, frame_360)
            context_events = pbp.build_context_window(events, anchor_idx, {"reason": reason})
            flattened = pbp.build_flattened_stream(
                context_events, anchor_event, {"reason": reason}, idx_360
            )
            prompt = _build_phase_buildup_prompt(flattened, intensity, anchor_event)
            line = pbp.generate_commentary(prompt)
        except Exception:
            line = ""
        if line:
            return pbp.enforce_surname_only(line, events).strip()

    return pbp.enforce_surname_only(
        pbp.generate_template_commentary(events, anchor_idx, {"reason": reason}),
        events,
    ).strip()


def build_climax_phase_timeline(events, threesixty_lookup, base_ts, climax_events, llm_available):
    """
    For each climax event, gather 3-5 uninterrupted buildup actions, stitch
    them with the climax call, and return one flowing PBP entry.
    """
    print("\n=== Building Climax Phase-of-Play Timeline ===")
    phase_entries = []
    subsumed_ids = set()

    if not climax_events:
        return phase_entries, subsumed_ids

    idx_360 = threesixty_lookup

    for cx in climax_events:
        cx_ts = cx["video_ts"]
        cx_event_id = cx["event_id"]
        cx_event = cx["event"]
        is_goal = cx["is_goal"]
        cx_idx = next(
            (i for i, ev in enumerate(events) if tac.get_event_id(ev) == cx_event_id), None
        )
        if cx_idx is None:
            continue

        lookback_start = cx_ts - CLIMAX_MIN_ANCHOR_LOOKBACK_SECONDS
        climax_possession = cx_event.get("possession")
        harvested = []

        for idx in range(cx_idx - 1, -1, -1):
            event = events[idx]
            event_ts = event_video_seconds(event, base_ts)
            if event_ts < lookback_start:
                break
            if is_climax_event(event):
                break
            if (
                climax_possession is not None
                and event.get("possession") is not None
                and event.get("possession") != climax_possession
            ):
                break
            if not is_phase_buildup_event(event):
                continue
            harvested.append((event_ts, idx, event))
            if len(harvested) >= CLIMAX_LOOKBACK_ANCHORS:
                break

        harvested.sort(key=lambda item: item[0])

        climax_line = ""
        if llm_available and pbp.is_llm_commentary_event(cx_event):
            frame_360 = idx_360.get(cx_event_id)
            intensity = pbp.classify_intensity(cx_event, frame_360)
            ctx_events = pbp.build_context_window(events, cx_idx, {"reason": "climax"})
            flattened = pbp.build_flattened_stream(
                ctx_events, cx_event, {"reason": "climax"}, idx_360
            )
            prompt = _build_climax_llm_prompt(flattened, intensity, cx_event, is_goal)
            climax_line = pbp.generate_commentary(prompt)
        if not climax_line:
            climax_line = pbp.generate_template_commentary(
                events, cx_idx, {"reason": "goal" if is_goal else "shot"}
            )
        climax_line = pbp.enforce_surname_only(climax_line, events).strip()
        if not climax_line:
            continue

        frame_360 = idx_360.get(cx_event_id)
        intensity = "Goal" if is_goal else pbp.classify_intensity(cx_event, frame_360)

        if len(harvested) < CLIMAX_MIN_BUILDUP_ANCHORS:
            direct_entry = {
                "video_ts": cx_ts + REACTION_BUFFER_SECONDS,
                "climax_ts": cx_ts,
                "commentator": "pbp",
                "text": climax_line,
                "event_id": f"climax_direct::{cx_event_id}",
                "selection_reason": "climax_direct",
                "is_goal": is_goal,
                "is_foul": False,
                "intensity": intensity,
                "event": cx_event,
                "is_climax_direct": True,
                "phase_event_ids": [],
                "phase_lines": [climax_line],
                "phase_line_items": [{"event_id": cx_event_id, "video_ts": cx_ts, "text": climax_line}],
            }
            phase_entries.append(direct_entry)
            subsumed_ids.add(cx_event_id)
            print(
                f"  [DIRECT CLIMAX] Climax @ {cx_ts:.1f}s | "
                f"only {len(harvested)} buildup anchors available."
            )
            continue

        commentary_lines = []
        phase_line_items = []
        phase_event_ids = set()
        t_start = harvested[0][0]

        for a_ts, anchor_idx, anchor_event in harvested:
            reason = "climax_phase_buildup"
            a_id = anchor_event.get("id")
            phase_event_ids.add(a_id)
            line = _generate_phase_anchor_line(
                events, idx_360, anchor_idx, anchor_event, reason, llm_available
            )
            if line:
                commentary_lines.append(line)
                phase_line_items.append({"event_id": a_id, "video_ts": a_ts, "text": line})

        if not commentary_lines:
            continue

        commentary_lines.append(climax_line)
        phase_line_items.append({"event_id": cx_event_id, "video_ts": cx_ts, "text": climax_line})
        stitched_text = stitch_pbp_lines(commentary_lines)

        phase_entry = {
            "video_ts": t_start,
            "climax_ts": cx_ts,
            "time_budget": max(0.0, cx_ts - t_start),
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
            "phase_lines": commentary_lines,
            "phase_line_items": phase_line_items,
        }
        phase_entries.append(phase_entry)
        subsumed_ids.update(phase_event_ids)
        subsumed_ids.add(cx_event_id)

        print(
            f"  [PHASE] Climax @ {cx_ts:.1f}s | buildup from {t_start:.1f}s "
            f"| {len(commentary_lines)} lines | '{stitched_text[:70]}...'"
        )

    return phase_entries, subsumed_ids


# ─────────────────────────────────────────────────────────────
# AUDIO BACK-ALIGNMENT
# ─────────────────────────────────────────────────────────────

def back_align_climax_entries(timeline):
    """
    Pin climax_phase audio end to T_climax + REACTION_BUFFER_SECONDS.
    Formula: T_start = T_climax - audio_duration + REACTION_BUFFER_SECONDS
    """
    print("\n=== Back-Aligning Climax Phase Audio ===")
    for entry in timeline:
        if not entry.get("is_climax_phase") or not entry.get("audio_path"):
            continue
        climax_ts = entry.get("climax_ts", entry["video_ts"])
        audio_dur = entry.get("audio_duration", 0.0)
        new_start = max(0.0, climax_ts - audio_dur + REACTION_BUFFER_SECONDS)
        old_start = entry["video_ts"]
        entry["video_ts"] = new_start
        print(
            f"  [BACK-ALIGN] climax={climax_ts:.1f}s dur={audio_dur:.1f}s "
            f"start: {old_start:.1f}s -> {new_start:.1f}s"
        )

    return resolve_audio_collisions(timeline)

# ─────────────────────────────────────────────────────────────
# PRIORITY ARBITRATION ENGINE
# ─────────────────────────────────────────────────────────────

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
    Run the tactical pipeline with dynamic cooldown and climax exclusion zones.
    """
    print("\n=== Running Tactical Commentary Pipeline ===")
    climax_timestamps = climax_timestamps or []
    commentary_plan = tac.build_commentary_plan(events, threesixty_lookup)

    if not commentary_plan:
        print("  No tactical events selected.")
        return []

    event_id_to_event = {}
    for ev in events:
        eid = tac.get_event_id(ev)
        if eid:
            event_id_to_event[eid] = ev

    entries = []
    spatial_count = 0
    last_spatial_ts = -TAC_SPATIAL_COOLDOWN_SECONDS
    clip_duration = tac.get_clip_duration_seconds(events)
    max_spatial_tac = max(1, int(clip_duration / TAC_SPATIAL_COOLDOWN_SECONDS))
    print(f"  Clip duration: {clip_duration:.1f}s | Max spatial TAC: {max_spatial_tac}")

    for item in commentary_plan:
        target_id = item["event_id"]
        event = event_id_to_event.get(target_id)
        if event is None:
            continue

        video_ts = event_video_seconds(event, base_ts)
        reason = item.get("selection_reason", "auto")
        is_mandatory = reason in ("goal", "foul")
        is_high_value_setpiece = (
            tac.is_corner_or_free_kick_event(event) and is_penalty_area_event(event)
        )

        # Climax exclusion zone
        if not is_mandatory:
            in_climax_zone = any(
                -TAC_CLIMAX_EXCLUSION_AFTER <= (video_ts - ct) <= TAC_CLIMAX_EXCLUSION_BEFORE
                for ct in climax_timestamps
            )
            if in_climax_zone:
                print(f"  [TAC SKIP - CLIMAX ZONE] {target_id} @ {video_ts:.1f}s")
                continue

        # Spatial cooldown
        is_spatial = reason.startswith("spatial_")
        if is_spatial and not is_high_value_setpiece:
            if spatial_count >= max_spatial_tac:
                print(f"  [TAC SKIP - MAX SPATIAL] {target_id} @ {video_ts:.1f}s")
                continue
            if (video_ts - last_spatial_ts) < TAC_SPATIAL_COOLDOWN_SECONDS:
                print(
                    f"  [TAC SKIP - COOLDOWN] {target_id} @ {video_ts:.1f}s "
                    f"(gap={video_ts - last_spatial_ts:.1f}s)"
                )
                continue

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


def _gap_fill_candidate_score(event):
    etype = tac.get_event_type(event)
    if not tac.get_event_id(event):
        return -100

    player = pbp.get_player_last_name(event)
    has_player = player and player not in {"Unknown", "Unknown Player", "Player"}
    score_by_type = {
        "Pass": 90,
        "Carry": 85,
        "Dribble": 82,
        "Interception": 80,
        "Ball Recovery": 78,
        "Goal Keeper": 74,
        "Duel": 70,
        "Miscontrol": 66,
        "Dispossessed": 64,
        "Ball Out": 58,
    }
    score = score_by_type.get(etype, 50 if "free-kick" in str(etype).lower() else 35)
    if has_player:
        score += 8
    if tac.get_event_location(event) is not None:
        score += 2
    return score


def _build_gap_fill_commentary(events, event_idx):
    event = events[event_idx]
    etype = tac.get_event_type(event)
    player = pbp.get_player_last_name(event)
    team = pbp.get_team_name(event)
    player_label = player if player and player not in {"Unknown", "Unknown Player"} else None
    team_label = team if team and team != "Unknown Team" else "the side in possession"

    if etype == "Duel":
        return f"{player_label} contests it." if player_label else "A challenge comes in."
    if etype == "Miscontrol":
        return f"{player_label} cannot quite bring it under control." if player_label else "The touch gets away."
    if etype == "Dispossessed":
        return f"{player_label} is crowded off it." if player_label else "Possession is disrupted."
    if etype == "Ball Out":
        return "The ball runs out of play."
    if "free-kick" in str(etype).lower():
        return f"{team_label} restart and look to build again."

    return pbp.generate_template_commentary(events, event_idx, {"reason": "Continuity Gap Fill"})


def _choose_gap_fill_event(events, base_ts, start_ts, end_ts, target_ts, used_ids):
    best = None
    for idx, event in enumerate(events):
        event_id = tac.get_event_id(event)
        if not event_id or event_id in used_ids:
            continue
        ts = event_video_seconds(event, base_ts)
        if ts < start_ts or ts > end_ts:
            continue
        score = _gap_fill_candidate_score(event) - abs(ts - target_ts) * 3.0
        if best is None or score > best[0]:
            best = (score, idx, event, ts)
    return best


def add_continuity_gap_fillers(timeline, events, threesixty_lookup, base_ts):
    """
    Add factual PBP continuity lines when the planned timeline has dead air.

    This pass happens after tactical/PBP arbitration, so it respects the current
    plan and only fills long empty stretches with real events from StatsBomb.
    Final duration-aware collision checks still run after TTS.
    """
    del threesixty_lookup
    if not events:
        return timeline

    timeline = sorted(timeline, key=lambda e: e["video_ts"])
    used_ids = {entry.get("event_id") for entry in timeline if entry.get("event_id")}
    fillers = []

    if not timeline:
        event_start = 0.0
        event_end = max(event_video_seconds(event, base_ts) for event in events)
        cursor_ts = event_start
        while event_end - cursor_ts > MAX_PLANNED_SILENCE_SECONDS:
            target_ts = cursor_ts + GAP_FILL_TARGET_SECONDS
            chosen = _choose_gap_fill_event(
                events,
                base_ts,
                cursor_ts + GAP_FILL_EDGE_BUFFER_SECONDS,
                min(event_end, target_ts + 3.0),
                target_ts,
                used_ids,
            )
            if not chosen:
                break
            _, event_idx, event, event_ts = chosen
            event_id = tac.get_event_id(event)
            text = pbp.enforce_surname_only(_build_gap_fill_commentary(events, event_idx), events).strip()
            if not text:
                break
            fillers.append({
                "video_ts": event_ts,
                "commentator": "pbp",
                "text": text,
                "event_id": event_id,
                "selection_reason": "continuity_gap_fill",
                "is_goal": pbp.is_goal_event(event),
                "is_foul": tac.is_foul_event(event),
                "intensity": "Neutral",
                "event": event,
                "is_gap_filler": True,
            })
            used_ids.add(event_id)
            cursor_ts = event_ts
        return sorted(fillers, key=lambda e: e["video_ts"])

    for prev, nxt in zip(timeline, timeline[1:]):
        cursor_ts = prev["video_ts"]
        next_ts = nxt["video_ts"]
        while next_ts - cursor_ts > MAX_PLANNED_SILENCE_SECONDS:
            target_ts = cursor_ts + GAP_FILL_TARGET_SECONDS
            window_start = cursor_ts + GAP_FILL_EDGE_BUFFER_SECONDS
            window_end = min(next_ts - GAP_FILL_EDGE_BUFFER_SECONDS, target_ts + 3.0)
            if window_end <= window_start:
                break

            chosen = _choose_gap_fill_event(
                events, base_ts, window_start, window_end, target_ts, used_ids
            )
            if not chosen:
                break

            _, event_idx, event, event_ts = chosen
            event_id = tac.get_event_id(event)
            text = pbp.enforce_surname_only(_build_gap_fill_commentary(events, event_idx), events).strip()
            if not text:
                break

            filler = {
                "video_ts": event_ts,
                "commentator": "pbp",
                "text": text,
                "event_id": event_id,
                "selection_reason": "continuity_gap_fill",
                "is_goal": pbp.is_goal_event(event),
                "is_foul": tac.is_foul_event(event),
                "intensity": "Neutral",
                "event": event,
                "is_gap_filler": True,
            }
            fillers.append(filler)
            used_ids.add(event_id)
            cursor_ts = event_ts
            print(
                f"  [GAP FILL] Added PBP at {event_ts:.1f}s between "
                f"{prev['video_ts']:.1f}s and {next_ts:.1f}s: {text}"
            )

    if fillers:
        timeline.extend(fillers)
        timeline.sort(key=lambda e: e["video_ts"])
        print(f"  Added {len(fillers)} continuity PBP line(s) to reduce silence.")
    return timeline


def merge_timelines(tac_entries, pbp_entries):
    """
    Merge tactical and PBP timelines with priority arbitration.

    Rules:
    1. No identical event_id at the same time by both commentators
    2. For spatial_midfield_dense_frame: PBP excluded 3s before, 5s after
    3. For corners/free kicks: PBP excluded ±3s around tactical event
    4. For goals/fouls: PBP speaks first, then tactical (delayed)
    5. For penalty area / shot events: PBP gets priority
    6. Tactical always gets general priority outside penalty area
    """
    print("\n=== Merging Timelines ===")

    # ── Step 1: Build exclusion zones from tactical events ──
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
            # Rule 6: corners/free kicks — tactical priority, exclude PBP nearby
            pbp_exclude_ranges.append((ts - SET_PIECE_EXCLUDE_RADIUS, ts + SET_PIECE_EXCLUDE_RADIUS))

    # ── Step 2: Handle goal/foul events — PBP first, tactical after ──
    for entry in tac_entries:
        if entry.get("is_goal") or entry.get("is_foul"):
            # Delay tactical commentary after PBP
            entry["video_ts"] = entry["video_ts"] + GOAL_FOUL_TAC_DELAY

    # ── Step 3: For penalty area / high-action events, PBP gets priority ──
    for entry in tac_entries:
        event = entry.get("event", {})
        if is_high_action_pbp_priority_event(event):
            reason = entry.get("selection_reason", "")
            # Don't suppress tactical for goal/foul (they're already delayed)
            if not entry.get("is_goal") and not entry.get("is_foul"):
                ts = entry["video_ts"]
                tac_exclude_ranges.append((ts - 2.0, ts + 2.0))

    # ── Step 4: Filter PBP entries by exclusion zones ──
    filtered_pbp = []
    for entry in pbp_entries:
        ts = entry["video_ts"]
        excluded = any(start <= ts <= end for start, end in pbp_exclude_ranges)
        if excluded:
            print(f"  [PBP EXCLUDED] {entry['event_id']} at {ts:.1f}s — "
                  f"within tactical exclusion zone")
            continue
        filtered_pbp.append(entry)

    # ── Step 5: Filter tactical entries in PBP priority zones ──
    filtered_tac = []
    for entry in tac_entries:
        ts = entry["video_ts"]
        excluded = any(start <= ts <= end for start, end in tac_exclude_ranges)
        if excluded:
            print(f"  [TAC EXCLUDED] {entry['event_id']} at {ts:.1f}s — "
                  f"PBP has priority (penalty area)")
            continue
        filtered_tac.append(entry)

    # ── Step 6: Combine and sort by timestamp ──
    combined = filtered_tac + filtered_pbp
    combined.sort(key=lambda e: e["video_ts"])

    # ── Step 7: Remove duplicates (same event_id) ──
    seen_ids = set()
    deduped = []
    for entry in combined:
        eid = entry["event_id"]
        if eid in seen_ids:
            print(f"  [DEDUP] Dropped duplicate event {eid}")
            continue
        seen_ids.add(eid)
        deduped.append(entry)

    # ── Step 8: Enforce minimum gap between entries ──
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


# ─────────────────────────────────────────────────────────────
# TEXT-TO-SPEECH
# ─────────────────────────────────────────────────────────────

def generate_tts_audio(timeline, progress_callback=None, events=None, threesixty_lookup=None, base_ts=0.0):
    """
    Generate WAV files for each commentary entry using Qwen3-TTS voice cloning.
    Uses Qwen/Qwen3-TTS-12Hz-0.6B-Base for voice cloning from reference audio samples.
    Returns list of entries with 'audio_path' and 'audio_duration' added.
    """
    print("\n=== Generating TTS Audio with Qwen3-TTS (0.6B) ===")

    import wave
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    # Reference text transcripts for each voice sample (needed for high-quality cloning)
    REF_TEXTS = {
        "Goal.mp3": (
            "Goal, Messi finds space in the box to head the ball. "
            "that was a brilliant effort by him and the teammates. "
            "Neymar with an amazing assist"
        ),
        "High.mp3": (
            "Messi is making a move towards the goal and he makes a pass to neymey, "
            "Neymar hit a volley, and Oh he misses it, that was a close one"
        ),
        "Neutral.mp3": (
            "England tactical formation seems decent, Belgium players are gathering "
            "around for a strategic move, the opposition seems to adapt to this properly as well"
        ),
        "Tactical.mp3": (
            "Goal, Messi finds space in the box to head the ball. "
            "that was a brilliant effort by him and the teammates. "
            "Neymar with an assist"
        ),
    }

    if progress_callback:
        progress_callback("Initializing Qwen3-TTS (this may take a moment)...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Use bfloat16 on CUDA for lower VRAM usage on 4GB GPU, float32 on CPU
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    print(f"  Loading Qwen3-TTS-0.6B-Base on {device} (dtype={dtype})...")

    # Initialize model
    try:
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            device_map=device,
            dtype=dtype,
        )
    except Exception as e:
        print(f"  Failed to initialize Qwen3-TTS: {e}")
        return timeline

    # Pre-build reusable voice clone prompts for each reference voice
    # This avoids re-extracting voice features for every commentary line
    if progress_callback:
        progress_callback("Building voice clone prompts from reference audio...")

    voice_prompts = {}
    for filename, ref_text in REF_TEXTS.items():
        ref_path = os.path.join(SCRIPT_DIR, "TTS", filename)
        if os.path.exists(ref_path):
            try:
                print(f"  Building voice prompt for {filename}...")
                voice_prompts[filename] = model.create_voice_clone_prompt(
                    ref_audio=ref_path,
                    ref_text=ref_text,
                )
            except Exception as e:
                print(f"  WARNING: Failed to build prompt for {filename}: {e}")
                # Fallback: try without ref_text (x_vector_only_mode)
                try:
                    voice_prompts[filename] = model.create_voice_clone_prompt(
                        ref_audio=ref_path,
                        ref_text="",
                        x_vector_only_mode=True,
                    )
                except Exception as e2:
                    print(f"  ERROR: Could not build any prompt for {filename}: {e2}")

    if not voice_prompts:
        print("  ERROR: No voice prompts could be built. Aborting TTS.")
        return timeline

    print(f"  Built {len(voice_prompts)} voice clone prompts.")

    def render_text_to_wav(text, ref_filename, wav_path):
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="English",
            voice_clone_prompt=voice_prompts[ref_filename],
        )
        sf.write(wav_path, wavs[0], sr)
        if not os.path.exists(wav_path):
            return None, 0.0
        return wav_path, get_wav_duration(wav_path)

    def fit_climax_phase_audio(entry, ref_filename, base_wav_path):
        lines = list(entry.get("phase_lines") or [entry["text"]])
        items = list(entry.get("phase_line_items") or [])
        if not lines:
            lines = [entry["text"]]

        attempt = 0
        while True:
            text = stitch_pbp_lines(lines) if len(lines) > 1 else lines[0].strip()
            entry["text"] = text
            wav_path = base_wav_path.replace(".wav", f"_fit{attempt}.wav")
            audio_path, duration = render_text_to_wav(text, ref_filename, wav_path)
            if not audio_path:
                return None, 0.0

            if not entry.get("is_climax_phase"):
                return audio_path, duration

            climax_ts = entry.get("climax_ts", entry.get("video_ts", 0.0))
            budget = max(0.0, climax_ts - entry.get("video_ts", climax_ts))
            entry["time_budget"] = budget

            if budget <= 0.05 or duration <= budget:
                return audio_path, duration

            required_ratio = duration / budget
            if required_ratio <= MAX_SPEEDUP_RATIO:
                stretched_path = apply_time_stretch(audio_path, required_ratio, TEMP_DIR)
                stretched_duration = get_wav_duration(stretched_path)
                if stretched_duration and stretched_duration <= budget + 0.05:
                    print(
                        f"      [BUDGET] sped up climax phase "
                        f"{required_ratio:.2f}x to fit {budget:.1f}s"
                    )
                    return stretched_path, stretched_duration

            if len(lines) > 1:
                dropped = lines.pop(0)
                if items:
                    items.pop(0)
                if len(lines) == 1:
                    entry["is_climax_phase"] = False
                    entry["is_climax_direct"] = True
                    entry["selection_reason"] = "climax_direct_budget_fallback"
                    entry["video_ts"] = climax_ts + REACTION_BUFFER_SECONDS
                    entry["time_budget"] = 0.0
                    print(
                        "      [BUDGET] dropped buildup and switched to direct "
                        f"climax call: '{dropped[:40]}...'"
                    )
                elif items:
                    entry["video_ts"] = items[0].get("video_ts", entry["video_ts"])
                    entry["time_budget"] = max(0.0, climax_ts - entry["video_ts"])
                    print(f"      [BUDGET] dropped first buildup line: '{dropped[:40]}...'")
                attempt += 1
                continue

            entry["is_climax_phase"] = False
            entry["is_climax_direct"] = True
            entry["selection_reason"] = "climax_direct_budget_fallback"
            entry["video_ts"] = entry.get("climax_ts", entry["video_ts"]) + REACTION_BUFFER_SECONDS
            entry["time_budget"] = 0.0
            return audio_path, duration

    def render_gap_filler_entry(event_idx, event, event_ts, filler_number, used_ids):
        event_id = tac.get_event_id(event)
        if not event_id or event_id in used_ids:
            return None

        text = pbp.enforce_surname_only(_build_gap_fill_commentary(events, event_idx), events).strip()
        if not text:
            return None

        wav_path = os.path.join(TEMP_DIR, f"tts_gap_{filler_number:03d}_pbp.wav")
        audio_path, duration = render_text_to_wav(text, "Neutral.mp3", wav_path)
        if not audio_path or duration <= 0.0:
            return None

        return {
            "video_ts": event_ts,
            "commentator": "pbp",
            "text": text,
            "event_id": event_id,
            "selection_reason": "post_tts_continuity_gap_fill",
            "is_goal": pbp.is_goal_event(event),
            "is_foul": tac.is_foul_event(event),
            "intensity": "Neutral",
            "event": event,
            "is_gap_filler": True,
            "audio_path": audio_path,
            "audio_duration": duration,
        }

    def add_post_tts_gap_fillers(current_timeline):
        """
        Fill real silence after WAV durations are known.

        Candidate fillers are always PBP and are only committed if their measured
        audio can fit between confirmed commentary clips without overlap.
        """
        if not events or "Neutral.mp3" not in voice_prompts:
            return current_timeline

        resolved = resolve_audio_collisions(current_timeline)
        active = [
            entry for entry in resolved
            if entry.get("audio_path") and entry.get("audio_duration", 0.0) > 0.0
        ]
        active.sort(key=lambda e: e["video_ts"])

        used_ids = {entry.get("event_id") for entry in resolved if entry.get("event_id")}
        fillers = []
        filler_number = 0
        event_end = max(event_video_seconds(event, base_ts) for event in events)
        boundaries = [{
            "video_ts": 0.0,
            "audio_duration": 0.0,
            "audio_path": "__virtual_start__",
            "event_id": "__virtual_start__",
        }]
        boundaries.extend(active)
        boundaries.append({
            "video_ts": event_end,
            "audio_duration": 0.0,
            "audio_path": "__virtual_end__",
            "event_id": "__virtual_end__",
        })
        boundaries.sort(key=lambda e: e["video_ts"])

        def try_fill_between(prev_end, next_start):
            nonlocal filler_number
            cursor_end = prev_end
            while next_start - cursor_end > MAX_SILENCE_SECONDS:
                earliest_start = cursor_end + MIN_AUDIO_GAP_SECONDS
                latest_event_ts = next_start - MIN_AUDIO_GAP_SECONDS
                if latest_event_ts <= earliest_start:
                    break

                target_ts = min(cursor_end + GAP_FILL_TARGET_SECONDS, latest_event_ts)
                chosen = _choose_gap_fill_event(
                    events,
                    base_ts,
                    earliest_start,
                    latest_event_ts,
                    target_ts,
                    used_ids,
                )
                if not chosen:
                    break

                _, event_idx, event, event_ts = chosen
                filler = render_gap_filler_entry(event_idx, event, event_ts, filler_number, used_ids)
                filler_number += 1
                if not filler:
                    used_ids.add(tac.get_event_id(event))
                    break

                earliest_audio_start = cursor_end + MIN_AUDIO_GAP_SECONDS
                latest_audio_start = next_start - filler["audio_duration"] - MIN_AUDIO_GAP_SECONDS
                if latest_audio_start < earliest_audio_start:
                    print(
                        f"  [POST-TTS GAP] Skipped filler at {event_ts:.1f}s; "
                        f"{filler['audio_duration']:.1f}s cannot fit in {next_start - cursor_end:.1f}s gap."
                    )
                    used_ids.add(filler["event_id"])
                    break

                filler["video_ts"] = min(max(event_ts, earliest_audio_start), latest_audio_start)
                fillers.append(filler)
                used_ids.add(filler["event_id"])
                cursor_end = filler["video_ts"] + filler["audio_duration"]
                print(
                    f"  [POST-TTS GAP] Added PBP at {filler['video_ts']:.1f}s "
                    f"({filler['audio_duration']:.1f}s) inside {prev_end:.1f}s-{next_start:.1f}s: "
                    f"{filler['text']}"
                )

        for prev, nxt in zip(boundaries, boundaries[1:]):
            prev_end = prev["video_ts"] + prev.get("audio_duration", 0.0)
            next_start = nxt["video_ts"]
            try_fill_between(prev_end, next_start)

        if fillers:
            resolved.extend(fillers)
            resolved = resolve_audio_collisions(resolved)
            print(f"  Added {len(fillers)} post-TTS continuity PBP line(s).")
        return resolved

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

        # Determine which reference voice to use
        if entry["commentator"] == "pbp":
            intensity = entry.get("intensity", "Neutral")
            if intensity == "Goal":
                ref_filename = "Goal.mp3"
            elif intensity == "High":
                ref_filename = "High.mp3"
            else:
                ref_filename = "Neutral.mp3"
        else:  # tactical
            ref_filename = "Tactical.mp3"

        # Fallback to Neutral if chosen prompt is missing
        if ref_filename not in voice_prompts:
            if "Neutral.mp3" in voice_prompts:
                ref_filename = "Neutral.mp3"
            else:
                # No usable prompt at all — skip this entry
                print(f"  [{i+1}/{total}] ERROR: no voice prompt available for {ref_filename}")
                entry["audio_path"] = None
                entry["audio_duration"] = 0.0
                continue

        try:
            print(f"  [{i+1}/{total}] {entry['commentator'].upper():9s} - Voice: {ref_filename}")
            if entry.get("is_climax_phase"):
                audio_path, duration = fit_climax_phase_audio(entry, ref_filename, wav_path)
            else:
                audio_path, duration = render_text_to_wav(text, ref_filename, wav_path)

            if audio_path:
                entry["audio_path"] = audio_path
                entry["audio_duration"] = duration
                print(f"      @ {entry['video_ts']:6.1f}s | {duration:.1f}s | {entry['text'][:60]}...")
            else:
                entry["audio_path"] = None
                entry["audio_duration"] = 0.0
        except Exception as e:
            print(f"  [{i+1}/{total}] ERROR generating/reading audio: {e}")
            entry["audio_path"] = None
            entry["audio_duration"] = 0.0

    timeline = back_align_climax_entries(timeline)
    timeline = add_post_tts_gap_fillers(timeline)

    # Free GPU memory after TTS generation is complete
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return timeline


# ─────────────────────────────────────────────────────────────
# AUDIO COMPOSITION & VIDEO OUTPUT
# ─────────────────────────────────────────────────────────────

def compose_video_with_commentary(timeline, progress_callback=None):
    """
    Overlay TTS commentary audio on the original video's crowd audio.
    Returns path to the final output video.
    """
    print("\n=== Composing Final Video ===")
    if progress_callback:
        progress_callback("Composing video with commentary...")
    timeline = resolve_audio_collisions(timeline)

    from moviepy import VideoFileClip, AudioFileClip
    from pydub import AudioSegment
    import imageio_ffmpeg
    # Point pydub to the ffmpeg bundled with imageio_ffmpeg
    AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.ffprobe = imageio_ffmpeg.get_ffmpeg_exe()

    video = VideoFileClip(VIDEO_FILE)
    video_duration = video.duration

    # Load original audio as pydub segment for mixing
    # First extract original audio to a temp wav
    original_audio_path = os.path.join(TEMP_DIR, "original_audio.wav")
    if video.audio is not None:
        video.audio.write_audiofile(original_audio_path, logger=None)
        crowd_audio = AudioSegment.from_wav(original_audio_path)
    else:
        # No audio in video — create silence
        crowd_audio = AudioSegment.silent(duration=int(video_duration * 1000))

    # Build the commentary overlay track
    commentary_track = AudioSegment.silent(duration=len(crowd_audio))
    last_commentary_end_ms = -int(MIN_AUDIO_GAP_SECONDS * 1000)

    for entry in timeline:
        if entry.get("drop_audio"):
            continue
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
            min_start_ms = last_commentary_end_ms + int(MIN_AUDIO_GAP_SECONDS * 1000)
            if position_ms < min_start_ms:
                print(
                    f"  [FINAL AUDIO GUARD] Moving overlay "
                    f"{position_ms / 1000:.1f}s -> {min_start_ms / 1000:.1f}s"
                )
                position_ms = min_start_ms
            if position_ms >= len(commentary_track):
                continue
            # Boost commentary volume slightly
            tts_audio = tts_audio + 3  # +3 dB
            commentary_track = commentary_track.overlay(tts_audio, position=position_ms)
            last_commentary_end_ms = position_ms + len(tts_audio)
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


# ─────────────────────────────────────────────────────────────
# VIDEO PLAYBACK (VLC embedded in Tkinter)
# ─────────────────────────────────────────────────────────────

def play_video_vlc(video_path, parent_window=None):
    """Play video using VLC embedded in tkinter, or fallback to system player."""
    try:
        import vlc
        # Create a new top-level window for the player
        player_win = tk.Toplevel(parent_window) if parent_window else tk.Tk()
        player_win.title("⚽ Commentary Playback")
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
                play_btn.config(text="▶ Play")
            else:
                player.play()
                play_btn.config(text="⏸ Pause")

        def on_stop():
            player.stop()
            player_win.destroy()

        play_btn = tk.Button(controls, text="⏸ Pause", command=on_play_pause,
                             bg="#e94560", fg="white", font=("Segoe UI", 11, "bold"),
                             relief=tk.FLAT, padx=20, pady=5)
        play_btn.pack(side=tk.LEFT, padx=10, pady=10)

        stop_btn = tk.Button(controls, text="✖ Close", command=on_stop,
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


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE (called from GUI)
# ─────────────────────────────────────────────────────────────

def run_pipeline(level, progress_callback=None, done_callback=None):
    """
    Full pipeline: load → generate both commentaries → merge → TTS → compose video.
    Runs in a background thread.
    """
    try:
        if progress_callback:
            progress_callback("Loading data...")

        events, threesixty_lookup = load_data()
        base_ts = get_base_timestamp(events)
        print(f"Base timestamp: {base_ts:.3f}s (first event)")

        # ── Ensure Ollama models are ready ──
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

        # ── Generate both commentary timelines ──
        if progress_callback:
            progress_callback("Generating tactical commentary...")
        # -- Identify climax events
        climax_events = identify_climax_events(events, base_ts)
        climax_timestamps = [cx["video_ts"] for cx in climax_events]
        llm_available = pbp.check_ollama_running() and pbp.check_model_exists(pbp.MODEL_NAME)
        if progress_callback:
            progress_callback("Building climax phase-of-play commentary...")
        phase_entries, subsumed_ids = build_climax_phase_timeline(
            events, threesixty_lookup, base_ts, climax_events, llm_available
        )
        tac_entries = build_tactical_timeline(
            events, threesixty_lookup, level, base_ts,
            climax_timestamps=climax_timestamps,
        )

        if progress_callback:
            progress_callback("Generating play-by-play commentary...")
        pbp_entries = build_pbp_timeline(events, threesixty_lookup, base_ts, subsumed_ids=subsumed_ids)
        pbp_entries.extend(phase_entries)

        # ── Merge with priority arbitration ──
        if progress_callback:
            progress_callback("Merging commentary timelines...")
        timeline = merge_timelines(tac_entries, pbp_entries)

        # ── Print unified timeline ──
        print("\n=== Unified Commentary Timeline ===")
        for i, entry in enumerate(timeline):
            tag = "🎯 TAC" if entry["commentator"] == "tactical" else "🎙️ PBP"
            print(f"  [{i+1:2d}] {entry['video_ts']:6.1f}s {tag} | {entry['text'][:80]}...")

        # ── Generate TTS audio ──
        if progress_callback:
            progress_callback("Generating voice audio...")
        timeline = generate_tts_audio(
            timeline,
            progress_callback,
            events=events,
            threesixty_lookup=threesixty_lookup,
            base_ts=base_ts,
        )

        # ── Compose final video ──
        if progress_callback:
            progress_callback("Building final video (this may take a moment)...")
        output_path = compose_video_with_commentary(timeline, progress_callback)

        if done_callback:
            done_callback(output_path)

    except Exception as e:
        import traceback
        traceback.print_exc()
        if done_callback:
            done_callback(None, error=str(e))


# ─────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────

class CommentaryApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⚽ Merged Commentary System")
        self.root.geometry("520x420")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f3460")

        self.output_video_path = None
        self._build_ui()

    def _build_ui(self):
        root = self.root

        # ── Title ──
        title_frame = tk.Frame(root, bg="#0f3460")
        title_frame.pack(pady=(25, 10))

        tk.Label(title_frame, text="⚽", font=("Segoe UI", 28),
                 bg="#0f3460", fg="white").pack()
        tk.Label(title_frame, text="Merged Commentary System",
                 font=("Segoe UI", 18, "bold"), bg="#0f3460", fg="#e94560").pack()
        tk.Label(title_frame, text="Tactical + Play-by-Play Commentary Generator",
                 font=("Segoe UI", 10), bg="#0f3460", fg="#a3a3c2").pack(pady=(2, 0))

        # ── Level Selection ──
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

        # ── Generate Button ──
        self.generate_btn = tk.Button(
            root, text="🎙️  Generate Commentary",
            command=self._on_generate,
            font=("Segoe UI", 12, "bold"), bg="#e94560", fg="white",
            relief=tk.FLAT, padx=20, pady=8, cursor="hand2",
        )
        self.generate_btn.pack(pady=(10, 5))

        # ── Progress Label ──
        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(
            root, textvariable=self.progress_var,
            font=("Segoe UI", 10), bg="#0f3460", fg="#a3a3c2",
        )
        self.progress_label.pack(pady=5)

        # ── Progress Bar ──
        self.progress_bar = ttk.Progressbar(root, mode="indeterminate", length=300)

        # ── Play Button (hidden until ready) ──
        self.play_btn = tk.Button(
            root, text="▶  Play Video with Commentary",
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
        self.progress_var.set("✅ Commentary generated successfully!")
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


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CommentaryApp()
    app.run()
