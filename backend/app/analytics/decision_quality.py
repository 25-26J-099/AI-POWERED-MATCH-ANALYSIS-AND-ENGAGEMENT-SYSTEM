"""Decision Quality (DQ) analytics — implements the full NB02–NB09 pipeline.

Coordinate convention (matches the research notebooks):
  All spatial features are normalised to [0, 1]:
    x_norm = x_pitch / 120,  y_norm = y_pitch / 80

  Event locations stored in raw_data are in StatsBomb pitch coords (0-120, 0-80).
  Freeze-frame player locations are pixel coords from the video frame and are
  converted:  x_norm = x_px / frame_w,  y_norm = y_px / frame_h
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Pitch / grid constants ─────────────────────────────────────────────────
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
FRAME_W = 1280.0
FRAME_H = 720.0

GOAL_X = 1.0
GOAL_Y = 0.5

# Analytical xT grid (16 × 12, normalised coords) — from NB08
_NX, _NY = 16, 12
_xc = (np.arange(_NX) + 0.5) / _NX
_yc = (np.arange(_NY) + 0.5) / _NY
_XX, _YY = np.meshgrid(_xc, _yc, indexing="ij")
_raw = (_XX ** 2.5) * np.exp(-((_YY - 0.5) ** 2) / (2 * 0.25 ** 2))
_XT_GRID: np.ndarray = 0.02 + (_raw / _raw.max()) * 0.33

# DQ hyper-parameters (NB08)
BEST_ALPHA = 0.15
GOAL_REWARD = 1.0
STAKE_SCALE = 10.0
MIN_CANDIDATES = 2

FEATURE_COLS = [
    "ball_x", "ball_y", "dist_to_goal", "angle_to_goal",
    "nearest_defender_dist", "num_defenders_close", "opponent_density",
    "defensive_compactness", "nearest_teammate_dist", "defenders_ahead",
    "type_dribble", "type_pass", "type_shot",
    "target_x", "target_y", "distance",
    "cand_nearest_def_dist", "cand_avg_top2_def_dist", "cand_num_defenders_near",
    "cand_num_defenders_in_lane", "cand_min_def_dist_to_lane", "cand_defenders_ahead",
]

VALID_EVENT_TYPES = {"Pass", "Carry", "Dribble", "Shot", "Ball Receipt*"}


# ── Coordinate helpers ─────────────────────────────────────────────────────

def _pitch_to_norm(x: float, y: float):
    return x / PITCH_LENGTH, y / PITCH_WIDTH


def _pixel_to_norm(x: float, y: float, fw: float = FRAME_W, fh: float = FRAME_H):
    return x / fw, y / fh


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _get_xt(x_norm: float, y_norm: float) -> float:
    if math.isnan(x_norm) or math.isnan(y_norm):
        return 0.0
    gx = min(max(int(x_norm * _NX), 0), _NX - 1)
    gy = min(max(int(y_norm * _NY), 0), _NY - 1)
    return float(_XT_GRID[gx, gy])


# ── Freeze-frame parsing ───────────────────────────────────────────────────

def _parse_freeze_players(freeze_frame, fw: float = FRAME_W, fh: float = FRAME_H) -> list:
    """Return list of dicts with normalised x/y and boolean flags."""
    if not freeze_frame:
        return []

    raw_list: list = []
    if isinstance(freeze_frame, dict):
        raw_list = freeze_frame.get("players", [])
    elif isinstance(freeze_frame, list):
        raw_list = freeze_frame

    players = []
    for p in raw_list:
        loc = p.get("location", [])
        if not loc or len(loc) < 2:
            continue
        xn, yn = _pixel_to_norm(float(loc[0]), float(loc[1]), fw, fh)
        players.append({
            "x": xn,
            "y": yn,
            "teammate": bool(p.get("teammate", False)),
            "actor": bool(p.get("actor", False)),
            "keeper": bool(p.get("keeper", False)),
            "player_id": p.get("player_id"),
        })
    return players


# ── NB02: State representation S ──────────────────────────────────────────

def _extract_state(x: float, y: float, players: list) -> dict:
    teammates = [p for p in players if p["teammate"] and not p["actor"]]
    opponents = [p for p in players if not p["teammate"]]

    dist_goal = _dist(x, y, GOAL_X, GOAL_Y)
    angle_goal = math.atan2(abs(GOAL_Y - y), abs(GOAL_X - x))

    def_dists = [_dist(x, y, o["x"], o["y"]) for o in opponents]
    nearest_def = min(def_dists) if def_dists else float("nan")
    num_close = sum(1 for d in def_dists if d < 0.05)
    opp_density = len(opponents)

    if opponents:
        cx = sum(o["x"] for o in opponents) / len(opponents)
        cy = sum(o["y"] for o in opponents) / len(opponents)
        compactness = sum(_dist(cx, cy, o["x"], o["y"]) for o in opponents) / len(opponents)
    else:
        compactness = float("nan")

    tm_dists = [_dist(x, y, t["x"], t["y"]) for t in teammates]
    nearest_tm = min(tm_dists) if tm_dists else float("nan")
    defenders_ahead = sum(1 for o in opponents if o["x"] > x)

    return {
        "ball_x": x, "ball_y": y,
        "dist_to_goal": dist_goal,
        "angle_to_goal": angle_goal,
        "nearest_defender_dist": nearest_def if math.isfinite(nearest_def) else float("nan"),
        "num_defenders_close": num_close,
        "opponent_density": opp_density,
        "defensive_compactness": compactness if math.isfinite(compactness) else float("nan"),
        "nearest_teammate_dist": nearest_tm if math.isfinite(nearest_tm) else float("nan"),
        "defenders_ahead": defenders_ahead,
    }


# ── NB03: Candidate generation S' ─────────────────────────────────────────

def _generate_candidates(x: float, y: float, players: list) -> list:
    candidates = []

    # Pass candidates — every visible teammate
    for p in players:
        if not p["teammate"] or p["actor"]:
            continue
        tx, ty = p["x"], p["y"]
        candidates.append({
            "type": "pass", "target_x": tx, "target_y": ty,
            "distance": _dist(x, y, tx, ty),
            "type_pass": True, "type_dribble": False, "type_shot": False,
        })

    # Dribble candidates — nearby grid cells (3 closest)
    GX, GY = 12, 8
    gx = int(x * GX)
    gy = int(y * GY)
    dribble_cands = []
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < GX and 0 <= ny < GY:
                tx = (nx + 0.5) / GX
                ty = (ny + 0.5) / GY
                dribble_cands.append({
                    "type": "dribble", "target_x": tx, "target_y": ty,
                    "distance": _dist(x, y, tx, ty),
                    "type_pass": False, "type_dribble": True, "type_shot": False,
                })
    dribble_cands.sort(key=lambda c: c["distance"])
    candidates.extend(dribble_cands[:3])

    # Shot candidate — attacking third only
    if x > 0.75:
        candidates.append({
            "type": "shot", "target_x": 1.0, "target_y": 0.5,
            "distance": _dist(x, y, 1.0, 0.5),
            "type_pass": False, "type_dribble": False, "type_shot": True,
        })

    return candidates


# ── NB05: Opponent features per candidate ─────────────────────────────────

def _point_to_segment_dist(px, py, x1, y1, x2, y2) -> float:
    mag_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if mag_sq < 1e-12:
        return _dist(px, py, x1, y1)
    u = max(0.0, min(1.0, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / mag_sq))
    return _dist(px, py, x1 + u * (x2 - x1), y1 + u * (y2 - y1))


def _compute_opponent_features(x: float, y: float, tx: float, ty: float, opponents: list) -> dict:
    if not opponents:
        return {
            "cand_nearest_def_dist": float("nan"),
            "cand_avg_top2_def_dist": float("nan"),
            "cand_num_defenders_near": 0,
            "cand_num_defenders_in_lane": 0,
            "cand_min_def_dist_to_lane": float("nan"),
            "cand_defenders_ahead": 0,
        }

    dists = sorted(_dist(tx, ty, o["x"], o["y"]) for o in opponents)
    nearest_def = dists[0]
    avg_top2 = sum(dists[:2]) / min(2, len(dists))
    num_local = sum(1 for d in dists if d < 0.1)

    lane_dists = [
        _point_to_segment_dist(o["x"], o["y"], x, y, tx, ty)
        for o in opponents
    ]
    in_lane = [d for d in lane_dists if d < 0.03]
    defenders_ahead = sum(1 for o in opponents if o["x"] > x)

    return {
        "cand_nearest_def_dist": nearest_def,
        "cand_avg_top2_def_dist": avg_top2,
        "cand_num_defenders_near": num_local,
        "cand_num_defenders_in_lane": len(in_lane),
        "cand_min_def_dist_to_lane": min(in_lane) if in_lane else float("nan"),
        "cand_defenders_ahead": defenders_ahead,
    }


# ── NB06: Feature vector construction ─────────────────────────────────────

_NAN_FILLS = {
    "cand_nearest_def_dist": 1.0,
    "cand_avg_top2_def_dist": 1.0,
    "cand_min_def_dist_to_lane": 1.0,
    "cand_num_defenders_near": 0.0,
    "cand_num_defenders_in_lane": 0.0,
    "cand_defenders_ahead": 0.0,
    "nearest_defender_dist": 0.0,
    "defensive_compactness": 0.0,
    "nearest_teammate_dist": 0.0,
}


def _build_feature_row(state: dict, cand: dict, opp: dict) -> list:
    raw = [
        state["ball_x"], state["ball_y"], state["dist_to_goal"], state["angle_to_goal"],
        state["nearest_defender_dist"], state["num_defenders_close"], state["opponent_density"],
        state["defensive_compactness"], state["nearest_teammate_dist"], state["defenders_ahead"],
        1.0 if cand["type_dribble"] else 0.0,
        1.0 if cand["type_pass"] else 0.0,
        1.0 if cand["type_shot"] else 0.0,
        cand["target_x"], cand["target_y"], cand["distance"],
        opp["cand_nearest_def_dist"], opp["cand_avg_top2_def_dist"],
        opp["cand_num_defenders_near"], opp["cand_num_defenders_in_lane"],
        opp["cand_min_def_dist_to_lane"], opp["cand_defenders_ahead"],
    ]
    result = []
    for i, v in enumerate(raw):
        if v != v or (isinstance(v, float) and math.isinf(v)):
            result.append(_NAN_FILLS.get(FEATURE_COLS[i], 0.0))
        else:
            result.append(float(v))
    return result


# ── NB04: Match chosen action to candidate ────────────────────────────────

def _match_chosen(ev: dict, candidates: list) -> Optional[int]:
    etype = ev["event_type"]

    if etype == "Shot":
        for i, c in enumerate(candidates):
            if c["type"] == "shot":
                return i
        return None

    if etype == "Pass" and ev["end_x"] is not None:
        best_i, best_d = None, float("inf")
        for i, c in enumerate(candidates):
            if c["type"] != "pass":
                continue
            d = _dist(ev["end_x"], ev["end_y"], c["target_x"], c["target_y"])
            if d < best_d:
                best_d, best_i = d, i
        return best_i if best_d < 0.05 else None

    if etype in ("Carry", "Dribble"):
        best_i, best_d = None, float("inf")
        for i, c in enumerate(candidates):
            if c["type"] != "dribble":
                continue
            d = _dist(ev["x"], ev["y"], c["target_x"], c["target_y"])
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    return None


# ── DQ score conversion (NB09) ────────────────────────────────────────────

def _dq_to_score(z: float, z_min: float, z_max: float,
                 target_min: float = 10.0, target_max: float = 92.0) -> float:
    z_range = max(z_max - z_min, 1e-6)
    scaled = (z - z_min) / z_range
    return round(scaled * (target_max - target_min) + target_min, 1)


def _tier(score: float) -> str:
    if score >= 83:
        return "Elite"
    if score >= 69:
        return "Very Good"
    if score >= 55:
        return "Good"
    if score >= 45:
        return "Average"
    if score >= 31:
        return "Below Average"
    return "Poor"


# ── Main pipeline ──────────────────────────────────────────────────────────

def compute_match_decision_quality(
    events,
    frame_w: float = FRAME_W,
    frame_h: float = FRAME_H,
) -> dict:
    """Compute Decision Quality for all players in a match.

    Parameters
    ----------
    events : list of Event ORM objects or raw event dicts.
    frame_w, frame_h : video frame dimensions used to convert pixel freeze-frame
        coordinates into normalised pitch coords.

    Returns
    -------
    dict with keys:
      players        — per-player DQ scores (0-100) and tier
      best_decisions — top 3 individual decisions (highest DQ)
      worst_decisions — bottom 3 individual decisions (lowest DQ)
      total_events_analyzed — number of events that contributed
    """
    from app.models.dq_model import predict_success_probability

    # ── Step 1: Parse raw events ───────────────────────────────────────────
    event_records: list[dict] = []

    for ev in events:
        raw: dict = ev.raw_data if hasattr(ev, "raw_data") else ev
        if not raw:
            continue

        etype_raw = raw.get("type", {})
        etype = etype_raw.get("name", "") if isinstance(etype_raw, dict) else str(etype_raw)
        if etype not in VALID_EVENT_TYPES:
            continue

        loc = raw.get("location", [])
        if not loc or len(loc) < 2:
            continue

        x_norm, y_norm = _pitch_to_norm(float(loc[0]), float(loc[1]))

        end_x_norm = end_y_norm = None
        for sub_key in ("pass", "carry", "shot"):
            sub = raw.get(sub_key, {})
            if isinstance(sub, dict):
                el = sub.get("end_location", [])
                if el and len(el) >= 2:
                    end_x_norm, end_y_norm = _pitch_to_norm(float(el[0]), float(el[1]))
                    break

        players = _parse_freeze_players(raw.get("freeze_frame"), frame_w, frame_h)
        if not players:
            continue

        shot_outcome = None
        if etype == "Shot":
            shot_sub = raw.get("shot", {})
            if isinstance(shot_sub, dict):
                oc = shot_sub.get("outcome", {})
                shot_outcome = oc.get("name") if isinstance(oc, dict) else None

        player_raw = raw.get("player", {})
        team_raw = raw.get("team", {})

        event_records.append({
            "event_uuid": raw.get("id", ""),
            "event_type": etype,
            "x": x_norm,
            "y": y_norm,
            "end_x": end_x_norm,
            "end_y": end_y_norm,
            "players": players,
            "shot_outcome": shot_outcome,
            "minute": int(raw.get("minute", 0)),
            "second": int(raw.get("second", 0)),
            "period": int(raw.get("period", 1)),
            "player_id": player_raw.get("id") if isinstance(player_raw, dict) else None,
            "player_name": (player_raw.get("name") or "Unknown") if isinstance(player_raw, dict) else "Unknown",
            "team": (team_raw.get("name") or "") if isinstance(team_raw, dict) else "",
        })

    if not event_records:
        return _empty_result()

    # ── Step 2: Outcome labelling (NB04) ──────────────────────────────────
    for i, ev in enumerate(event_records):
        if ev["event_type"] == "Shot":
            ev["outcome"] = "success" if ev["shot_outcome"] == "Goal" else "failure"
        elif i < len(event_records) - 1:
            nxt = event_records[i + 1]
            ev["outcome"] = "success" if nxt["team"] == ev["team"] else "failure"
        else:
            ev["outcome"] = "success"

    # ── Step 3: State + candidates + features ─────────────────────────────
    all_rows: list[dict] = []

    for ev_idx, ev in enumerate(event_records):
        x, y = ev["x"], ev["y"]
        players = ev["players"]
        opponents = [p for p in players if not p["teammate"]]

        state = _extract_state(x, y, players)
        candidates = _generate_candidates(x, y, players)

        if len(candidates) < MIN_CANDIDATES:
            continue

        chosen_idx = _match_chosen(ev, candidates)

        for c_idx, cand in enumerate(candidates):
            opp_feats = _compute_opponent_features(x, y, cand["target_x"], cand["target_y"], opponents)
            feat_row = _build_feature_row(state, cand, opp_feats)

            all_rows.append({
                "event_idx": ev_idx,
                "candidate_idx": c_idx,
                "is_chosen": (c_idx == chosen_idx),
                "feature_row": feat_row,
                "xT_start": _get_xt(x, y),
                "xT_target": _get_xt(cand["target_x"], cand["target_y"]),
                "cand_type": cand["type"],
            })

    if not all_rows:
        return _empty_result()

    # ── Step 4: Predict success probabilities (NB07) ──────────────────────
    X = np.array([r["feature_row"] for r in all_rows], dtype=np.float64)
    p_success = predict_success_probability(X)
    for j, row in enumerate(all_rows):
        row["p_success"] = float(p_success[j])

    # ── Step 5: Compute V per candidate (NB08) ────────────────────────────
    event_groups: dict[int, list[int]] = defaultdict(list)
    for j, row in enumerate(all_rows):
        event_groups[row["event_idx"]].append(j)

    for ev_idx, row_indices in event_groups.items():
        ps = [all_rows[j]["p_success"] for j in row_indices]
        mean_p = sum(ps) / len(ps)
        std_p = (sum((p - mean_p) ** 2 for p in ps) / len(ps)) ** 0.5

        ev = event_records[ev_idx]
        for j in row_indices:
            row = all_rows[j]
            xT_gain = row["xT_target"] - row["xT_start"]
            reward = xT_gain
            if row["is_chosen"] and ev.get("shot_outcome") == "Goal":
                reward = GOAL_REWARD

            action_weight = 1.0 + row["xT_start"] * STAKE_SCALE
            p_rel = (row["p_success"] - mean_p) / (std_p + 1e-6)
            row.update({
                "xT_gain": xT_gain,
                "reward": reward,
                "action_weight": action_weight,
                "p_relative": p_rel,
                "V": reward + BEST_ALPHA * p_rel,
            })

    # ── Step 6: Compute DQ per event (NB08) ──────────────────────────────
    dq_records: list[dict] = []

    for ev_idx, row_indices in event_groups.items():
        chosen_rows = [j for j in row_indices if all_rows[j]["is_chosen"]]
        if not chosen_rows:
            continue

        chosen_j = chosen_rows[0]
        chosen_row = all_rows[chosen_j]
        n_cand = len(row_indices)

        V_vals = [all_rows[j]["V"] for j in row_indices]
        best_V = max(V_vals)
        worst_V = min(V_vals)
        chosen_V = chosen_row["V"]

        dq_raw = chosen_V - best_V
        dq_norm = (dq_raw / (best_V - worst_V)) if best_V != worst_V else 0.0
        dq_adj = dq_raw / math.log(n_cand + 1)

        ev = event_records[ev_idx]
        dq_records.append({
            "event_uuid": ev["event_uuid"],
            "player_id": ev["player_id"],
            "player_name": ev["player_name"],
            "team": ev["team"],
            "minute": ev["minute"],
            "second": ev["second"],
            "period": ev["period"],
            "event_type": ev["event_type"],
            "action_type": chosen_row["cand_type"],
            "dq_raw": dq_raw,
            "dq_norm": dq_norm,
            "dq_adj": dq_adj,
            "action_weight": chosen_row["action_weight"],
            "n_candidates": n_cand,
            "chosen_V": chosen_V,
            "best_V": best_V,
        })

    if not dq_records:
        return _empty_result()

    # ── Step 7: Player-level aggregation (NB08/NB09) ──────────────────────
    player_buckets: dict = defaultdict(list)
    for rec in dq_records:
        key = rec["player_id"] or rec["player_name"]
        player_buckets[key].append(rec)

    player_stats: list[dict] = []
    for key, recs in player_buckets.items():
        dq_vals = [r["dq_adj"] for r in recs]
        weights = [r["action_weight"] for r in recs]
        total_w = sum(weights)
        weighted_dq = sum(d * w for d, w in zip(dq_vals, weights)) / (total_w or 1)
        player_stats.append({
            "player_id": recs[0]["player_id"],
            "player_name": recs[0]["player_name"],
            "team": recs[0]["team"],
            "weighted_dq": weighted_dq,
            "mean_dq": sum(dq_vals) / len(dq_vals),
            "n_actions": len(recs),
            "pct_optimal": sum(1 for d in dq_vals if d >= -1e-6) / len(dq_vals),
        })

    # 0-100 score (NB09)
    dq_list = [p["weighted_dq"] for p in player_stats]
    z_min, z_max = min(dq_list), max(dq_list)
    for p in player_stats:
        p["dq_score"] = _dq_to_score(p["weighted_dq"], z_min, z_max) if len(player_stats) > 1 else 50.0
        p["tier"] = _tier(p["dq_score"])

    player_stats.sort(key=lambda p: p["dq_score"], reverse=True)

    # ── Step 8: Best / worst decisions ────────────────────────────────────
    sorted_dq = sorted(dq_records, key=lambda r: r["dq_adj"], reverse=True)
    best_3 = [_fmt_decision(r) for r in sorted_dq[:3]]
    worst_3 = [_fmt_decision(r) for r in sorted_dq[-3:]]

    return {
        "players": [
            {
                "player_id": p["player_id"],
                "player_name": p["player_name"],
                "team": p["team"],
                "dq_score": p["dq_score"],
                "tier": p["tier"],
                "weighted_dq": round(p["weighted_dq"], 4),
                "n_actions": p["n_actions"],
                "pct_optimal": round(p["pct_optimal"] * 100, 1),
            }
            for p in player_stats
        ],
        "best_decisions": best_3,
        "worst_decisions": worst_3,
        "total_events_analyzed": len(dq_records),
    }


def _fmt_decision(r: dict) -> dict:
    action = r["action_type"].capitalize()
    time_str = f"P{r['period']} {r['minute']:02d}:{r['second']:02d}"
    score = round(r["dq_adj"], 4)
    if score >= -0.001:
        quality = "optimal"
    elif score >= -0.05:
        quality = "near-optimal"
    elif score >= -0.15:
        quality = "suboptimal"
    else:
        quality = "poor"
    return {
        "event_uuid": r["event_uuid"],
        "player_name": r["player_name"],
        "team": r["team"],
        "minute": r["minute"],
        "second": r["second"],
        "period": r["period"],
        "event_type": r["event_type"],
        "action_type": r["action_type"],
        "dq_score": score,
        "description": f"{action} at {time_str} — {quality} decision",
    }


def _empty_result() -> dict:
    return {
        "players": [],
        "best_decisions": [],
        "worst_decisions": [],
        "total_events_analyzed": 0,
    }
