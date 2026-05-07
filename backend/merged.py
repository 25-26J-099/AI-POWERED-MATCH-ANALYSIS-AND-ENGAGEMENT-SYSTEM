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
import glob
import json
import math
import os
import random
import re
import shutil
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

# -- Dynamic Tactical Cooldown
TAC_SPATIAL_COOLDOWN_SECONDS = 50.0    # Min gap between spatial TAC commentary events
TAC_CLIMAX_EXCLUSION_BEFORE = 11.0    # Block TAC N seconds before a climax event
TAC_CLIMAX_EXCLUSION_AFTER  = 2.0     # Block TAC N seconds after a climax event

# -- Climax Phase-of-Play (PBP Stitching)
CLIMAX_LOOKBACK_ANCHORS = 5           # Max anchors to stitch before a climax event
CLIMAX_PHASE_SEPARATOR = '... '       # Separator between stitched PBP lines
CLIMAX_MIN_ANCHOR_LOOKBACK_SECONDS = 20.0

# -- Audio Time-Budgeting & Back-Alignment
REACTION_BUFFER_SECONDS = 0.75         # Delay after climax before commentary lands
MAX_SPEEDUP_RATIO = 1.25               # Never stretch audio faster than 1.25x
MIN_AUDIO_GAP_SECONDS = 0.30           # Minimum gap between any two audio clips

TEMP_DIR = os.path.join(tempfile.gettempdir(), "merged_commentary")
os.makedirs(TEMP_DIR, exist_ok=True)


def _prepend_binary_dir(binary_path):
    if not binary_path or not os.path.isfile(binary_path):
        return None
    binary_dir = os.path.dirname(os.path.abspath(binary_path))
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if binary_dir not in path_parts:
        os.environ["PATH"] = binary_dir + os.pathsep + os.environ.get("PATH", "")
    return os.path.abspath(binary_path)


def _resolve_binary(executable_name, env_name, candidate_paths=None):
    configured = _prepend_binary_dir(os.environ.get(env_name))
    if configured:
        return configured

    found = shutil.which(executable_name)
    if found:
        _prepend_binary_dir(found)
        return found

    for candidate in candidate_paths or []:
        for path in glob.glob(os.path.expandvars(os.path.expanduser(candidate))):
            resolved = _prepend_binary_dir(path)
            if resolved:
                return resolved
    return None


def configure_external_audio_tools():
    """Make SoX/ffmpeg discoverable even when Uvicorn starts from a stale PATH."""
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    sox_candidates = [
        os.path.join(local_appdata, "Microsoft", "WinGet", "Packages", "ChrisBagwell.SoX_*", "sox-*", "sox.exe"),
        r"C:\Program Files*\sox*\sox.exe",
    ]
    ffmpeg_candidates = [
        os.path.join(SCRIPT_DIR, ".venv", "Lib", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg*.exe"),
        os.path.join(SCRIPT_DIR, ".venv", "lib", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg*.exe"),
    ]

    sox_path = _resolve_binary("sox", "SOX_BINARY", sox_candidates)

    ffmpeg_path = os.environ.get("IMAGEIO_FFMPEG_EXE") or _resolve_binary("ffmpeg", "FFMPEG_BINARY", ffmpeg_candidates)
    if not ffmpeg_path:
        try:
            import imageio_ffmpeg

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_path = None
    if ffmpeg_path:
        _prepend_binary_dir(ffmpeg_path)
        if os.path.basename(ffmpeg_path).lower() != "ffmpeg.exe":
            shim_dir = os.path.join(TEMP_DIR, "bin")
            os.makedirs(shim_dir, exist_ok=True)
            shim_path = os.path.join(shim_dir, "ffmpeg.exe")
            try:
                if not os.path.exists(shim_path) or os.path.getsize(shim_path) != os.path.getsize(ffmpeg_path):
                    shutil.copy2(ffmpeg_path, shim_path)
                ffmpeg_path = shim_path
            except Exception:
                pass
            _prepend_binary_dir(ffmpeg_path)
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path
        os.environ["FFMPEG_BINARY"] = ffmpeg_path

    return sox_path, ffmpeg_path


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
    allowed_names = sorted(pbp.collect_allowed_names([anchor_event]))
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
    if is_goal:
        parts.append(
            "[SYSTEM INSTRUCTION: This is a GOAL. Generate a short, explosive, "
            "celebratory exclamation. Use dramatic language. Keep it under 15 words.]"
        )
    else:
        parts.append(
            "[SYSTEM INSTRUCTION: This is a climax moment (shot/chance). "
            "Generate a short, punchy call connecting to the build-up. "
            "Do NOT use GOAL. Keep it under 12 words.]"
        )
    parts.append("")
    parts.append("Flattened Event Stream:")
    parts.append(flattened_stream)
    return "\n".join(parts)


def build_climax_phase_timeline(events, threesixty_lookup, base_ts, climax_events, llm_available):
    """
    For each climax event, gather preceding PBP anchors, stitch them together,
    and return a single climax_phase entry per climax.
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

        lookback_start = cx_ts - CLIMAX_MIN_ANCHOR_LOOKBACK_SECONDS
        harvested = []
        for anchor in all_anchors:
            a_ts = event_video_seconds(anchor["event"], base_ts)
            a_id = anchor["event"].get("id")
            if a_id == cx_event_id or is_climax_event(anchor["event"]):
                continue
            if lookback_start <= a_ts < cx_ts:
                harvested.append((a_ts, anchor))

        harvested.sort(key=lambda x: x[0])
        harvested = harvested[-CLIMAX_LOOKBACK_ANCHORS:]

        commentary_lines = []
        phase_event_ids = set()
        t_start = cx_ts

        for a_ts, anchor in harvested:
            anchor_idx = anchor["index"]
            anchor_event = anchor["event"]
            reason = anchor.get("reason", "climax_phase_buildup")
            a_id = anchor_event.get("id")
            phase_event_ids.add(a_id)
            line = pbp.generate_template_commentary(events, anchor_idx, {"reason": reason})
            line = pbp.ensure_allowed_commentary_names(line, events, anchor_idx, {"reason": reason})
            if line:
                commentary_lines.append(line)
            if a_ts < t_start:
                t_start = a_ts

        cx_idx = next(
            (i for i, ev in enumerate(events) if tac.get_event_id(ev) == cx_event_id), None
        )
        if cx_idx is not None:
            if llm_available and pbp.is_llm_commentary_event(cx_event):
                frame_360 = idx_360.get(cx_event_id)
                intensity = pbp.classify_intensity(cx_event, frame_360)
                ctx_events = pbp.build_context_window(events, cx_idx, {"reason": "climax"})
                flattened = pbp.build_flattened_stream(
                    ctx_events, cx_event, {"reason": "climax"}, idx_360
                )
                prompt = _build_climax_llm_prompt(flattened, intensity, cx_event, is_goal)
                climax_line = pbp.generate_commentary(prompt)
                climax_line = pbp.ensure_allowed_commentary_names(
                    climax_line,
                    events,
                    cx_idx,
                    {"reason": "goal" if is_goal else "shot"},
                )
            else:
                climax_line = pbp.generate_template_commentary(
                    events, cx_idx, {"reason": "goal" if is_goal else "shot"}
                )
                climax_line = pbp.ensure_allowed_commentary_names(
                    climax_line,
                    events,
                    cx_idx,
                    {"reason": "goal" if is_goal else "shot"},
                )
            if climax_line:
                commentary_lines.append(climax_line)

        if not commentary_lines:
            continue

        stitched_text = stitch_pbp_lines(commentary_lines)
        frame_360 = idx_360.get(cx_event_id)
        intensity = "Goal" if is_goal else pbp.classify_intensity(cx_event, frame_360)

        phase_entry = {
            "video_ts": t_start,
            "climax_ts": cx_ts,
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

    timeline.sort(key=lambda e: e["video_ts"])
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]
        if not prev.get("audio_path") or not curr.get("audio_path"):
            continue
        prev_end = prev["video_ts"] + prev.get("audio_duration", 0.0)
        if curr["video_ts"] < prev_end + MIN_AUDIO_GAP_SECONDS:
            if curr.get("is_climax_phase"):
                print(f"  [COLLISION] Climax phase @ {curr['video_ts']:.1f}s wins over prev.")
            else:
                curr["video_ts"] = prev_end + MIN_AUDIO_GAP_SECONDS

    return timeline

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
    deferred_climax_zone_items = []
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
                if reason.startswith("spatial_"):
                    deferred_climax_zone_items.append(item)
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
        if not pbp.commentary_uses_only_allowed_names(commentary, events):
            commentary = tac.build_spatial_commentary_fallback(
                level,
                result.get("tactical_labels") or {},
                team_name=result.get("team"),
            )

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

    if not entries and deferred_climax_zone_items:
        item = deferred_climax_zone_items[0]
        target_id = item["event_id"]
        event = event_id_to_event.get(target_id)
        if event is not None:
            video_ts = event_video_seconds(event, base_ts)
            reason = item.get("selection_reason", "spatial_midfield_dense_frame")
            print(f"  [TAC FORCE - SHORT CLIP] {target_id} @ {video_ts:.1f}s")
            result = tac.process_event(
                event,
                events,
                threesixty_lookup,
                selection_reason=reason,
                source_event_id=item.get("source_event_id"),
            )
            commentary = tac.generate_commentary_ollama(
                result["tactical_description"],
                level,
                selection_reason=result.get("selection_reason", "auto"),
                tactical_labels=result.get("tactical_labels"),
                team_name=result.get("team"),
            )
            if not pbp.commentary_uses_only_allowed_names(commentary, events):
                commentary = tac.build_spatial_commentary_fallback(
                    level,
                    result.get("tactical_labels") or {},
                    team_name=result.get("team"),
                )
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
            commentary = pbp.ensure_allowed_commentary_names(
                commentary,
                events,
                anchor_idx,
                {"reason": reason},
            )
        else:
            commentary = pbp.generate_template_commentary(events, anchor_idx, {"reason": reason})
            commentary = pbp.ensure_allowed_commentary_names(
                commentary,
                events,
                anchor_idx,
                {"reason": reason},
            )

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
        if excluded and (entry.get("is_climax_phase") or entry.get("is_goal") or entry.get("is_foul")):
            excluded = False
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

    if not filtered_tac and tac_entries:
        forced = dict(tac_entries[0])
        forced["video_ts"] = max(0.0, forced["video_ts"] - 1.5)
        print(f"  [TAC FORCE - MERGE] Keeping tactical entry {forced['event_id']} for coverage")
        filtered_tac.append(forced)

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

def generate_tts_audio(timeline, progress_callback=None):
    """
    Generate WAV files for each commentary entry using Qwen3-TTS voice cloning.
    Uses Qwen/Qwen3-TTS-12Hz-0.6B-Base for voice cloning from reference audio samples.
    Returns list of entries with 'audio_path' and 'audio_duration' added.
    """
    print("\n=== Generating TTS Audio with Qwen3-TTS (0.6B) ===")

    import wave
    import torch
    import soundfile as sf
    configure_external_audio_tools()
    import contextlib
    import io

    # qwen_tts prints a flash-attn advisory on Windows; CUDA still runs with PyTorch SDPA.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
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
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="English",
                voice_clone_prompt=voice_prompts[ref_filename],
            )

            # Save the generated audio
            sf.write(wav_path, wavs[0], sr)

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

    # Free GPU memory after TTS generation is complete
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Fix overlaps: if one commentary would overlap the next, shift the next
    for i in range(1, len(timeline)):
        prev = timeline[i - 1]
        curr = timeline[i]
        if prev.get("audio_path") and curr.get("audio_path"):
            prev_end = prev["video_ts"] + prev["audio_duration"]
            if curr["video_ts"] < prev_end + 0.3:
                curr["video_ts"] = prev_end + 0.3

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

    _, ffmpeg_path = configure_external_audio_tools()
    from moviepy import VideoFileClip, AudioFileClip
    from pydub import AudioSegment
    import imageio_ffmpeg
    # Point pydub to the ffmpeg bundled with imageio_ffmpeg
    ffmpeg_path = ffmpeg_path or imageio_ffmpeg.get_ffmpeg_exe()
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffprobe = ffmpeg_path

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
        timeline = generate_tts_audio(timeline, progress_callback)

        # ── Compose final video ──
        if progress_callback:
            progress_callback("Building final video (this may take a moment)...")
        # -- Back-align climax audio
        timeline = back_align_climax_entries(timeline)

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
