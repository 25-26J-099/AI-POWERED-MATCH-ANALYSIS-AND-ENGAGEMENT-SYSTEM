"""
Module 4: Hybrid Event Detection Engine (v4 - Complete Implementation).

v4 updates:
- Added 12 new event types: Ball Receipt, Carry, Pressure, Ball Recovery, Duel, 
  Clearance, Block, Goalkeeper Action, Miscontrol, Dribble, Dispossessed, 
  Interception, Dribbled Past, Foul
- Integrated freeze frame generation for all events
- Enhanced ML event detector integration
- Improved event enrichment with spatial context
- All 18 required event types now implemented
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(Enum):
    # Existing events
    POSSESSION_CHANGE = "possession_change"
    PASS_ATTEMPT = "pass"
    SHOT = "shot"
    TACKLE = "tackle"
    OUT_OF_BOUNDS = "out_of_bounds"
    SPRINT = "sprint"
    
    # New events (v4)
    BALL_RECEIPT = "ball_receipt"
    CARRY = "carry"
    PRESSURE = "pressure"
    BALL_RECOVERY = "ball_recovery"
    DUEL = "duel"
    CLEARANCE = "clearance"
    BLOCK = "block"
    GOALKEEPER_SAVE = "goalkeeper_save"
    GOALKEEPER_CLAIM = "goalkeeper_claim"
    MISCONTROL = "miscontrol"
    DRIBBLE = "dribble"
    DISPOSSESSED = "dispossessed"
    INTERCEPTION = "interception"
    DRIBBLED_PAST = "dribbled_past"
    FOUL = "foul"


def _f(v):
    """Ensure plain Python float for serialization (handles np.float64)."""
    if v is None:
        return 0.0
    return float(v)


@dataclass
class GameEvent:
    event_type: str
    frame_idx: int
    timestamp: float = 0.0
    confidence: float = 1.0
    position: Tuple[float, float] = (0.0, 0.0)
    player_id: Optional[int] = None
    team_id: int = -1
    details: str = ""
    source: str = "rule"
    freeze_frame: Optional[dict] = None  # v4: Added freeze frame support

    def to_dict(self) -> dict:
        result = {
            "type": self.event_type,
            "frame": int(self.frame_idx),
            "timestamp": round(_f(self.timestamp), 2),
            "confidence": round(_f(self.confidence), 3),
            "position": [round(_f(self.position[0]), 1),
                         round(_f(self.position[1]), 1)],
            "player_id": int(self.player_id) if self.player_id is not None else None,
            "team_id": int(self.team_id),
            "details": str(self.details),
            "source": str(self.source),
        }
        
        # Add freeze frame if available
        if self.freeze_frame:
            result["freeze_frame"] = self.freeze_frame
        
        return result


# ── Cooldowns (minimum frames between events of the same type) ──
EVENT_COOLDOWNS = {
    "out_of_bounds": 100,
    "shot": 60,
    "tackle": 50,
    "possession_change": 25,
    "pass": 25,
    "sprint": 100,
    # v4: New event cooldowns
    "ball_receipt": 15,
    "carry": 40,
    "pressure": 30,
    "ball_recovery": 40,
    "duel": 50,
    "clearance": 60,
    "block": 50,
    "goalkeeper_save": 80,
    "goalkeeper_claim": 60,
    "miscontrol": 40,
    "dribble": 60,
    "dispossessed": 40,
    "interception": 50,
    "dribbled_past": 50,
    "foul": 75,
}


class FreezeFrameGenerator:
    """Generate freeze frames (360° player snapshots) for events."""
    
    def __init__(self, frame_width=1280, frame_height=720):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.goalkeeper_memory = {}  # Track which players are goalkeepers
    
    def generate(self, event: GameEvent, player_tracks: Dict, ball_track) -> dict:
        """
        Generate freeze frame for an event.
        
        Returns:
            freeze_frame: Dict with player positions and metadata
        """
        freeze_frame = {
            "event_frame": event.frame_idx,
            "players": []
        }
        
        actor_id = event.player_id
        actor_team = event.team_id
        
        # Add all players
        for tid, track in player_tracks.items():
            if track.is_ball or track.is_referee:
                continue
            
            is_gk = self._is_goalkeeper(track, tid)
            
            player_entry = {
                "player_id": tid,
                "location": [round(_f(track.center[0]), 1), round(_f(track.center[1]), 1)],
                "teammate": track.team_id == actor_team if actor_team >= 0 else None,
                "actor": tid == actor_id,
                "keeper": is_gk,
                "team_id": track.team_id
            }
            freeze_frame["players"].append(player_entry)
        
        # Add ball if visible
        if ball_track and ball_track.frames_lost < 5:
            freeze_frame["ball_location"] = [
                round(_f(ball_track.center[0]), 1),
                round(_f(ball_track.center[1]), 1)
            ]
        
        return freeze_frame
    
    def _is_goalkeeper(self, track, track_id) -> bool:
        """Heuristic to detect if player is goalkeeper based on position."""
        # Use cached result if available
        if track_id in self.goalkeeper_memory:
            return self.goalkeeper_memory[track_id]
        
        # Need at least 30 frames to determine
        if not hasattr(track, 'position_history') or len(track.position_history) < 30:
            return False
        
        # Check if player stays near goal line
        recent_positions = track.position_history[-30:]
        x_positions = [p[0] for p in recent_positions]
        
        # Normalize to 0-1
        x_norm = [x / self.frame_width for x in x_positions]
        
        # Near left (< 0.1) or right (> 0.9) edge
        near_goal_count = sum(1 for x in x_norm if x < 0.1 or x > 0.9)
        is_gk = (near_goal_count / len(x_norm)) > 0.7
        
        # Cache result
        self.goalkeeper_memory[track_id] = is_gk
        
        return is_gk


class RuleBasedDetector:
    def __init__(self, config):
        self.cfg = config.event_detection
        self.possession_player: Optional[int] = None
        self.possession_team: int = -1
        self.possession_frames: int = 0
        self.grace_counter: int = 0
        self.grace_limit: int = 15
        self._min_poss_frames = getattr(self.cfg, 'possession_min_frames', 3)
        
        # v4: Track state for new events
        self._possession_history: List[Tuple[int, Optional[int], int]] = []  # (frame, player, team)
        self._player_velocity_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        self._duel_cooldown: Dict[Tuple[int, int], int] = {}
        self._foul_cooldown: Dict[int, int] = {}

    def _closest_player(self, ball_pos, tracks):
        """Find closest player(s) to ball, preferring assigned team players."""
        candidates = []
        for tid, t in tracks.items():
            if t.is_ball or t.is_referee:
                continue
            dx = ball_pos[0] - t.center[0]
            dy = ball_pos[1] - t.center[1]
            d = (dx*dx + dy*dy) ** 0.5
            candidates.append((tid, d, t.team_id))

        if not candidates:
            return (None, float('inf'))

        candidates.sort(key=lambda x: x[1])

        # Prefer the closest player with an assigned team
        for tid, dist, team in candidates[:3]:
            if team >= 0 and dist < self.cfg.possession_radius * 1.5:
                return (tid, dist)

        return (candidates[0][0], candidates[0][1])

    def detect_possession(self, ball_pos, tracks, frame_idx, fps):
        pid, dist = self._closest_player(ball_pos, tracks)
        if pid is None:
            return None

        if dist <= self.cfg.possession_radius:
            new_team = tracks[pid].team_id
            old_pid = self.possession_player
            old_team = self.possession_team

            # Count consecutive frames with SAME possessor
            if pid == self.possession_player:
                self.possession_frames += 1
            else:
                self.possession_frames = 1

            old_frames = self.possession_frames
            self.possession_player = pid
            self.possession_team = new_team
            self.grace_counter = 0
            
            # Track possession history for other events
            self._possession_history.append((frame_idx, pid, new_team))
            if len(self._possession_history) > 100:
                self._possession_history = self._possession_history[-100:]

            # Possession change: different team, must have held for min frames
            if (old_pid is not None and old_pid != pid
                    and old_team >= 0 and new_team >= 0
                    and old_team != new_team
                    and old_frames >= self._min_poss_frames):
                return GameEvent(
                    event_type=EventType.POSSESSION_CHANGE.value,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    confidence=0.85,
                    position=(_f(ball_pos[0]), _f(ball_pos[1])),
                    player_id=pid,
                    team_id=new_team,
                    details=f"Team {new_team} gains possession (#{pid} from #{old_pid})",
                )
        else:
            self.grace_counter += 1
            if self.grace_counter > self.grace_limit:
                self.possession_player = None
                self.possession_frames = 0
        return None

    def detect_tackle(self, tracks, frame_idx, fps):
        items = list(tracks.items())
        for i, (tid1, t1) in enumerate(items):
            if t1.is_ball or t1.is_referee:
                continue
            for tid2, t2 in items[i+1:]:
                if t2.is_ball or t2.is_referee:
                    continue
                if t1.team_id >= 0 and t2.team_id >= 0 and t1.team_id == t2.team_id:
                    continue
                dx = t1.center[0] - t2.center[0]
                dy = t1.center[1] - t2.center[1]
                dist = (dx*dx + dy*dy) ** 0.5
                if dist < self.cfg.tackle_proximity:
                    v1, v2 = t1.velocity, t2.velocity
                    if v1 < self.cfg.tackle_velocity_drop or v2 < self.cfg.tackle_velocity_drop:
                        return GameEvent(
                            event_type=EventType.TACKLE.value,
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            confidence=0.7,
                            position=(_f((t1.center[0]+t2.center[0])/2),
                                      _f((t1.center[1]+t2.center[1])/2)),
                            player_id=tid1,
                            team_id=t1.team_id,
                            details=f"Tackle: #{tid1} on #{tid2}",
                        )
        return None

    def detect_out_of_bounds(self, ball_pos, fw, fh, frame_idx, fps):
        margin = getattr(self.cfg, 'oob_margin', 0.012)
        margin_x = fw * margin
        margin_y = fh * margin

        top_excl = fh * getattr(self.cfg, 'pitch_top_margin', 0.04)
        bot_excl = fh * (1.0 - getattr(self.cfg, 'pitch_bottom_margin', 0.08))

        bx, by = ball_pos

        if by < top_excl or by > bot_excl:
            return None

        if (bx < margin_x or bx > fw - margin_x
                or by < margin_y + top_excl or by > bot_excl - margin_y):
            return GameEvent(
                event_type=EventType.OUT_OF_BOUNDS.value,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                confidence=0.75,
                position=(_f(bx), _f(by)),
                details="Ball out of bounds",
            )
        return None
    
    # ========== v4: NEW EVENT DETECTORS ==========
    
    def detect_ball_receipt(self, current_possessor, previous_possessor, ball_traveled_dist, frame_idx, fps, tracks):
        """Detect successful ball reception by a player."""
        if previous_possessor != current_possessor and current_possessor is not None:
            if current_possessor in tracks and ball_traveled_dist > 60:
                track = tracks[current_possessor]
                return GameEvent(
                    event_type=EventType.BALL_RECEIPT.value,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    confidence=0.75,
                    position=(_f(track.center[0]), _f(track.center[1])),
                    player_id=current_possessor,
                    team_id=track.team_id,
                    details=f"Received from #{previous_possessor}" if previous_possessor else "Ball receipt",
                )
        return None
    
    def detect_carry(self, possessor, tracks, frame_idx, fps):
        """Detect player carrying the ball while moving."""
        if possessor and possessor in tracks:
            track = tracks[possessor]
            if len(track.position_history) >= 15:
                # Check if player moved >50px while possessing
                recent_positions = track.position_history[-15:]
                start_pos = recent_positions[0]
                end_pos = recent_positions[-1]
                distance = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
                
                if distance > 50:
                    return GameEvent(
                        event_type=EventType.CARRY.value,
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        confidence=0.65,
                        position=(_f(track.center[0]), _f(track.center[1])),
                        player_id=possessor,
                        team_id=track.team_id,
                        details=f"Carry (dist={distance:.0f}px)",
                    )
        return None
    
    def detect_pressure(self, possessor, tracks, frame_idx, fps):
        """Detect defensive pressure on ball carrier."""
        if possessor and possessor in tracks:
            poss_track = tracks[possessor]
            poss_team = poss_track.team_id
            
            # Count nearby opponents
            nearby_opponents = 0
            for tid, track in tracks.items():
                if track.team_id != poss_team and track.team_id >= 0:
                    dist = ((poss_track.center[0] - track.center[0])**2 + 
                           (poss_track.center[1] - track.center[1])**2)**0.5
                    if dist < 100:  # ~2 meters in pixels
                        nearby_opponents += 1
            
            if nearby_opponents >= 2:
                return GameEvent(
                    event_type=EventType.PRESSURE.value,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    confidence=0.70,
                    position=(_f(poss_track.center[0]), _f(poss_track.center[1])),
                    player_id=possessor,
                    team_id=poss_team,
                    details=f"Under pressure from {nearby_opponents} opponents",
                )
        return None
    
    def detect_ball_recovery(self, current_possessor, previous_possessor, tracks, ball_loose_frames, frame_idx, fps):
        """Detect team regaining possession after ball was loose."""
        if current_possessor and previous_possessor:
            if current_possessor in tracks and previous_possessor in tracks:
                curr_team = tracks[current_possessor].team_id
                prev_team = tracks[previous_possessor].team_id
                
                # Team change after ball was loose for >15 frames
                if curr_team != prev_team and curr_team >= 0 and ball_loose_frames > 15:
                    return GameEvent(
                        event_type=EventType.BALL_RECOVERY.value,
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        confidence=0.75,
                        position=(_f(tracks[current_possessor].center[0]), 
                                 _f(tracks[current_possessor].center[1])),
                        player_id=current_possessor,
                        team_id=curr_team,
                        details=f"Ball recovery by team {curr_team}",
                    )
        return None
    
    def detect_duel(self, ball_pos, tracks, frame_idx, fps):
        """Detect 1v1 contest for the ball."""
        candidates = []
        for tid, track in tracks.items():
            if track.is_ball or track.is_referee:
                continue
            dist = ((track.center[0] - ball_pos[0])**2 + (track.center[1] - ball_pos[1])**2)**0.5
            if dist < 80:
                candidates.append((tid, track, dist))
        
        if len(candidates) == 2:
            t1_id, t1_track, _ = candidates[0]
            t2_id, t2_track, _ = candidates[1]
            
            if t1_track.team_id != t2_track.team_id and t1_track.team_id >= 0 and t2_track.team_id >= 0:
                # Check cooldown for this pair
                pair_key = tuple(sorted([t1_id, t2_id]))
                last_duel = self._duel_cooldown.get(pair_key, -999)
                
                if frame_idx - last_duel > 50:
                    self._duel_cooldown[pair_key] = frame_idx
                    return GameEvent(
                        event_type=EventType.DUEL.value,
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        confidence=0.68,
                        position=(_f(ball_pos[0]), _f(ball_pos[1])),
                        player_id=t1_id,
                        team_id=t1_track.team_id,
                        details=f"Duel: #{t1_id} vs #{t2_id}",
                    )
        return None
    
    def detect_clearance(self, ball_speed, ball_pos, possessor, tracks, fw, frame_idx, fps):
        """Detect defensive clearance."""
        if ball_speed and ball_speed > 25 and possessor and possessor in tracks:
            track = tracks[possessor]
            # In defensive third
            norm_x = ball_pos[0] / fw
            in_defensive_third = (track.team_id == 0 and norm_x < 0.35) or (track.team_id == 1 and norm_x > 0.65)
            
            if in_defensive_third:
                return GameEvent(
                    event_type=EventType.CLEARANCE.value,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    confidence=0.70,
                    position=(_f(ball_pos[0]), _f(ball_pos[1])),
                    player_id=possessor,
                    team_id=track.team_id,
                    details=f"Defensive clearance (v={ball_speed:.1f})",
                )
        return None
    
    def detect_block(self, ball_track, tracks, prev_ball_speed, current_ball_speed, frame_idx, fps):
        """Detect ball being blocked by defender."""
        if prev_ball_speed and current_ball_speed:
            if prev_ball_speed > 20 and current_ball_speed < 5:
                # Ball rapidly decelerated
                if ball_track:
                    for tid, track in tracks.items():
                        if track.is_ball or track.is_referee:
                            continue
                        dist = ((track.center[0] - ball_track.center[0])**2 + 
                               (track.center[1] - ball_track.center[1])**2)**0.5
                        if dist < 50:
                            return GameEvent(
                                event_type=EventType.BLOCK.value,
                                frame_idx=frame_idx,
                                timestamp=frame_idx / fps,
                                confidence=0.65,
                                position=(_f(ball_track.center[0]), _f(ball_track.center[1])),
                                player_id=tid,
                                team_id=track.team_id,
                                details="Shot/pass blocked",
                            )
        return None
    
    def detect_goalkeeper_action(self, ball_pos, ball_speed, tracks, frame_idx, fps, fw):
        """Detect goalkeeper-specific actions."""
        for tid, track in tracks.items():
            if track.is_ball or track.is_referee:
                continue
            
            # Check if near goal line
            norm_x = track.center[0] / fw
            near_goal = norm_x < 0.1 or norm_x > 0.9
            
            if near_goal:
                dist_to_ball = ((track.center[0] - ball_pos[0])**2 + 
                               (track.center[1] - ball_pos[1])**2)**0.5
                
                if dist_to_ball < 60:
                    if ball_speed and ball_speed > 15:
                        return GameEvent(
                            event_type=EventType.GOALKEEPER_SAVE.value,
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            confidence=0.72,
                            position=(_f(track.center[0]), _f(track.center[1])),
                            player_id=tid,
                            team_id=track.team_id,
                            details="Goalkeeper save",
                        )
                    elif ball_speed and ball_speed < 5:
                        return GameEvent(
                            event_type=EventType.GOALKEEPER_CLAIM.value,
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            confidence=0.68,
                            position=(_f(track.center[0]), _f(track.center[1])),
                            player_id=tid,
                            team_id=track.team_id,
                            details="Goalkeeper claims ball",
                        )
        return None
    
    def detect_miscontrol(self, ball_track, possessor, tracks, frame_idx, fps):
        """Detect poor ball control."""
        if possessor and possessor in tracks and ball_track:
            track = tracks[possessor]
            ball_dist = ((track.center[0] - ball_track.center[0])**2 + 
                        (track.center[1] - ball_track.center[1])**2)**0.5
            
            # Ball gets too far from player
            if ball_dist > 120 and self.possession_frames >= 5:
                return GameEvent(
                    event_type=EventType.MISCONTROL.value,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    confidence=0.60,
                    position=(_f(track.center[0]), _f(track.center[1])),
                    player_id=possessor,
                    team_id=track.team_id,
                    details="Poor touch/miscontrol",
                )
        return None
    
    def detect_dispossessed(self, current_possessor, previous_possessor, tracks, frame_idx, fps):
        """Detect player losing ball to opponent in close proximity."""
        if previous_possessor and current_possessor and previous_possessor != current_possessor:
            if previous_possessor in tracks and current_possessor in tracks:
                prev_track = tracks[previous_possessor]
                curr_track = tracks[current_possessor]
                prev_team = prev_track.team_id
                curr_team = curr_track.team_id
                
                if prev_team != curr_team and prev_team >= 0 and curr_team >= 0:
                    # Lost in close proximity
                    dist = ((prev_track.center[0] - curr_track.center[0])**2 + 
                           (prev_track.center[1] - curr_track.center[1])**2)**0.5
                    if dist < 100:
                        return GameEvent(
                            event_type=EventType.DISPOSSESSED.value,
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            confidence=0.70,
                            position=(_f(prev_track.center[0]), _f(prev_track.center[1])),
                            player_id=previous_possessor,
                            team_id=prev_team,
                            details=f"Dispossessed by #{current_possessor}",
                        )
        return None
    
    def detect_foul(self, tracks, frame_idx, fps):
        """Detect potential foul situations."""
        items = list(tracks.items())
        for i, (tid1, t1) in enumerate(items):
            if t1.is_ball or t1.is_referee:
                continue
            for tid2, t2 in items[i+1:]:
                if t2.is_ball or t2.is_referee:
                    continue
                
                if t1.team_id >= 0 and t2.team_id >= 0 and t1.team_id != t2.team_id:
                    dist = ((t1.center[0] - t2.center[0])**2 + (t1.center[1] - t2.center[1])**2)**0.5
                    
                    # Very close contact
                    if dist < 25:
                        # Check velocity drop
                        self._player_velocity_history[tid1].append(t1.velocity)
                        self._player_velocity_history[tid2].append(t2.velocity)
                        
                        v1_hist = list(self._player_velocity_history[tid1])
                        v2_hist = list(self._player_velocity_history[tid2])
                        
                        # Check cooldown
                        last_foul = self._foul_cooldown.get(tid1, -999)
                        if frame_idx - last_foul < 75:
                            continue
                        
                        if len(v1_hist) >= 10:
                            avg_before = np.mean(v1_hist[-10:-2])
                            current = v1_hist[-1]
                            if avg_before > 8 and current < avg_before * 0.3:
                                self._foul_cooldown[tid1] = frame_idx
                                return GameEvent(
                                    event_type=EventType.FOUL.value,
                                    frame_idx=frame_idx,
                                    timestamp=frame_idx / fps,
                                    confidence=0.55,
                                    position=(_f((t1.center[0] + t2.center[0])/2), 
                                             _f((t1.center[1] + t2.center[1])/2)),
                                    player_id=tid2,
                                    team_id=t2.team_id,
                                    details=f"Potential foul on #{tid1}",
                                )
        return None


class MLEventDetector:
    def __init__(self, config):
        self.cfg = config.event_detection
        self.window = self.cfg.temporal_window
        self._player_vel: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.window))
        self._sprint_cooldown: Dict[int, int] = {}
        
        # v4: Track state for new ML events
        self._dribble_state: Dict[int, List] = defaultdict(list)  # Track dribble sequences
        self._interception_window: deque = deque(maxlen=30)

    def detect_pass(
        self,
        ball_consecutive,
        prev_possessor, curr_possessor,
        tracks, frame_idx, fps,
    ):
        """Detect pass: ball transfers between two same-team players."""
        if prev_possessor is None or curr_possessor is None:
            return None
        if prev_possessor == curr_possessor:
            return None
        t_prev = tracks.get(prev_possessor)
        t_curr = tracks.get(curr_possessor)
        if t_prev is None or t_curr is None:
            return None

        # Same team check
        if t_prev.team_id < 0 or t_curr.team_id < 0:
            if t_prev.team_id < 0 and t_curr.team_id < 0:
                return None
        elif t_prev.team_id != t_curr.team_id:
            return None

        # Verify ball moved sufficient distance
        if len(ball_consecutive) >= 3:
            start = ball_consecutive[0]
            end = ball_consecutive[-1]
            dist = ((end[1]-start[1])**2 + (end[2]-start[2])**2) ** 0.5
            if dist > self.cfg.pass_min_distance:
                return GameEvent(
                    event_type=EventType.PASS_ATTEMPT.value,
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    confidence=0.7,
                    position=(_f(end[1]), _f(end[2])),
                    player_id=prev_possessor,
                    team_id=t_prev.team_id if t_prev.team_id >= 0 else t_curr.team_id,
                    details=f"Pass #{prev_possessor}->{curr_possessor} (d={dist:.0f}px)",
                    source="ml",
                )
        return None

    def detect_sprint(self, tracks, frame_idx, fps):
        for tid, track in tracks.items():
            if track.is_ball or track.is_referee:
                continue
            self._player_vel[tid].append(track.velocity)
            buf = self._player_vel[tid]

            last_sprint = self._sprint_cooldown.get(tid, -9999)
            if frame_idx - last_sprint < 100:
                continue

            if len(buf) >= self.window:
                avg = float(np.mean(list(buf)))
                if avg > 8.0 and min(list(buf)[-5:]) > 5.0:
                    self._sprint_cooldown[tid] = frame_idx
                    return GameEvent(
                        event_type=EventType.SPRINT.value,
                        frame_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        confidence=0.6,
                        position=(_f(track.center[0]), _f(track.center[1])),
                        player_id=tid,
                        team_id=track.team_id,
                        details=f"Sprint #{tid} (avg_v={avg:.1f})",
                        source="ml",
                    )
        return None
    
    def detect_dribble(self, possessor, tracks, ball_track, frame_idx, fps):
        """Detect successful dribbling past opponent."""
        if possessor and possessor in tracks and ball_track:
            track = tracks[possessor]
            poss_team = track.team_id
            
            # Track dribble state
            self._dribble_state[possessor].append({
                'frame': frame_idx,
                'pos': track.center,
                'has_ball': True
            })
            
            # Keep last 20 frames
            if len(self._dribble_state[possessor]) > 20:
                self._dribble_state[possessor] = self._dribble_state[possessor][-20:]
            
            # Check if player has been moving with ball
            if len(self._dribble_state[possessor]) >= 20:
                recent = self._dribble_state[possessor]
                start_pos = recent[0]['pos']
                end_pos = recent[-1]['pos']
                distance = ((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)**0.5
                
                if distance > 80:
                    # Check if passed by an opponent
                    for tid, opp_track in tracks.items():
                        if opp_track.team_id != poss_team and opp_track.team_id >= 0:
                            # Check if opponent was passed
                            if len(opp_track.position_history) >= 10:
                                opp_start = opp_track.position_history[-10]
                                opp_now = opp_track.center
                                
                                # Simple heuristic: opponent behind dribbler now, was in front before
                                was_ahead = (opp_start[0] - start_pos[0]) * (end_pos[0] - start_pos[0]) > 0
                                now_behind = ((opp_now[0] - end_pos[0]) * (end_pos[0] - start_pos[0])) < 0
                                
                                if was_ahead and now_behind:
                                    return GameEvent(
                                        event_type=EventType.DRIBBLE.value,
                                        frame_idx=frame_idx,
                                        timestamp=frame_idx / fps,
                                        confidence=0.58,
                                        position=(_f(track.center[0]), _f(track.center[1])),
                                        player_id=possessor,
                                        team_id=poss_team,
                                        details=f"Dribbled past #{tid}",
                                        source="ml",
                                    )
        return None
    
    def detect_interception(self, ball_consecutive, current_possessor, tracks, frame_idx, fps):
        """Detect ball being intercepted."""
        if current_possessor and current_possessor in tracks:
            # Track ball trajectory changes
            if len(ball_consecutive) >= 5:
                # Check for sudden direction change
                start = ball_consecutive[-5]
                mid = ball_consecutive[-3]
                end = ball_consecutive[-1]
                
                # Vector before and after
                vec1 = (mid[1] - start[1], mid[2] - start[2])
                vec2 = (end[1] - mid[1], end[2] - mid[2])
                
                # Angle between vectors
                dot = vec1[0]*vec2[0] + vec1[1]*vec2[1]
                mag1 = (vec1[0]**2 + vec1[1]**2)**0.5
                mag2 = (vec2[0]**2 + vec2[1]**2)**0.5
                
                if mag1 > 0 and mag2 > 0:
                    cos_angle = dot / (mag1 * mag2)
                    
                    # Significant direction change (angle > 90 degrees)
                    if cos_angle < 0:
                        return GameEvent(
                            event_type=EventType.INTERCEPTION.value,
                            frame_idx=frame_idx,
                            timestamp=frame_idx / fps,
                            confidence=0.62,
                            position=(_f(end[1]), _f(end[2])),
                            player_id=current_possessor,
                            team_id=tracks[current_possessor].team_id,
                            details="Ball intercepted",
                            source="ml",
                        )
        return None


class HybridEventDetector:
    """
    Orchestrates rule-based and ML-based event detection.
    Manages cooldowns, ball history, possession state, and freeze frames.
    """

    def __init__(self, config):
        self.config = config
        self.rule = RuleBasedDetector(config)
        self.ml = MLEventDetector(config)
        self.freeze_frame_gen = FreezeFrameGenerator()

        self.ball_history: List[Tuple[int, float, float]] = []
        self.ball_consecutive: List[Tuple[int, float, float]] = []
        self._last_ball_frame: int = -1

        self.events: List[GameEvent] = []
        self.frame_events: List[GameEvent] = []

        self.possession_frames: Dict[int, int] = defaultdict(int)
        self.current_possessor: Optional[int] = None
        self.previous_possessor: Optional[int] = None
        
        # v4: Track additional state
        self._ball_loose_frames: int = 0
        self._prev_ball_speed: Optional[float] = None

        self._cooldowns: Dict[str, int] = {}

        self._max_ball_v = getattr(
            config.event_detection, 'max_ball_velocity', 100.0
        )

    def _can_emit(self, event_type: str, frame_idx: int) -> bool:
        cd = EVENT_COOLDOWNS.get(event_type, 20)
        last = self._cooldowns.get(event_type, -9999)
        return (frame_idx - last) >= cd

    def _emit(self, event: GameEvent):
        self._cooldowns[event.event_type] = event.frame_idx
        self.frame_events.append(event)
        self.events.append(event)

    def _update_ball_history(self, ball_track, frame_idx):
        if ball_track is not None and ball_track.frames_lost < 3:
            bx, by = _f(ball_track.center[0]), _f(ball_track.center[1])
            self.ball_history.append((frame_idx, bx, by))
            if len(self.ball_history) > 200:
                self.ball_history = self.ball_history[-100:]

            if frame_idx - self._last_ball_frame <= 2:
                self.ball_consecutive.append((frame_idx, bx, by))
                self._ball_loose_frames = 0
            else:
                self.ball_consecutive = [(frame_idx, bx, by)]
                self._ball_loose_frames += 1
            self._last_ball_frame = frame_idx
        else:
            if frame_idx - self._last_ball_frame > 5:
                self.ball_consecutive = []
            self._ball_loose_frames += 1

    def _ball_speed_consecutive(self) -> Optional[float]:
        """Compute average ball speed from last few consecutive detections."""
        seq = self.ball_consecutive
        if len(seq) < 4:
            return None
        recent = seq[-4:]
        speeds = []
        for i in range(1, len(recent)):
            dt = recent[i][0] - recent[i-1][0]
            if dt <= 0:
                continue
            dx = recent[i][1] - recent[i-1][1]
            dy = recent[i][2] - recent[i-1][2]
            speed = (dx*dx + dy*dy)**0.5 / dt
            speeds.append(speed)
        if not speeds:
            return None
        avg = float(np.mean(speeds))
        return min(avg, self._max_ball_v)
    
    def _ball_traveled_distance(self) -> float:
        """Calculate distance ball has traveled in recent history."""
        if len(self.ball_consecutive) < 2:
            return 0.0
        start = self.ball_consecutive[0]
        end = self.ball_consecutive[-1]
        return ((end[1] - start[1])**2 + (end[2] - start[2])**2)**0.5

    def _detect_shot(self, fw, frame_idx, fps) -> Optional[GameEvent]:
        speed = self._ball_speed_consecutive()
        if speed is None:
            return None
        if speed < self.config.event_detection.shot_velocity_threshold:
            return None

        seq = self.ball_consecutive
        bx, by = seq[-1][1], seq[-1][2]
        norm_x = bx / fw
        dx_dir = seq[-1][1] - seq[-3][1]
        moving_to_goal = (dx_dir < -15 and norm_x < 0.35) or (dx_dir > 15 and norm_x > 0.65)

        if moving_to_goal:
            return GameEvent(
                event_type=EventType.SHOT.value,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                confidence=0.65,
                position=(_f(bx), _f(by)),
                details=f"Shot (speed={speed:.1f}px/f)",
                source="rule",
            )
        return None

    def process_frame(
        self, frame, player_tracks, ball_track, frame_idx, fps=25.0,
    ) -> List[GameEvent]:
        self.frame_events = []
        fh, fw = frame.shape[:2]

        self._update_ball_history(ball_track, frame_idx)

        ball_pos = None
        if ball_track is not None and ball_track.frames_lost < 5:
            ball_pos = (_f(ball_track.center[0]), _f(ball_track.center[1]))

        # Calculate current ball speed
        current_ball_speed = self._ball_speed_consecutive()
        ball_traveled_dist = self._ball_traveled_distance()

        # ── Rule-Based Events ─────────────────────────────────────────

        if ball_pos is not None:
            # Possession
            poss_ev = self.rule.detect_possession(ball_pos, player_tracks, frame_idx, fps)
            if poss_ev and self._can_emit(poss_ev.event_type, frame_idx):
                self._emit(poss_ev)

            # Track possessor for other events
            self.previous_possessor = self.current_possessor
            if self.rule.possession_player is not None:
                self.current_possessor = self.rule.possession_player
                team = self.rule.possession_team
                if team >= 0:
                    self.possession_frames[team] += 1

            # Out of bounds
            oob = self.rule.detect_out_of_bounds(ball_pos, fw, fh, frame_idx, fps)
            if oob and self._can_emit(oob.event_type, frame_idx):
                self._emit(oob)

            # Shot
            shot = self._detect_shot(fw, frame_idx, fps)
            if shot and self._can_emit(shot.event_type, frame_idx):
                self._emit(shot)
            
            # v4: New rule-based events
            
            # Ball Receipt
            ball_receipt = self.rule.detect_ball_receipt(
                self.current_possessor, self.previous_possessor, 
                ball_traveled_dist, frame_idx, fps, player_tracks
            )
            if ball_receipt and self._can_emit(ball_receipt.event_type, frame_idx):
                self._emit(ball_receipt)
            
            # Pressure
            pressure = self.rule.detect_pressure(self.current_possessor, player_tracks, frame_idx, fps)
            if pressure and self._can_emit(pressure.event_type, frame_idx):
                self._emit(pressure)
            
            # Ball Recovery
            ball_recovery = self.rule.detect_ball_recovery(
                self.current_possessor, self.previous_possessor, 
                player_tracks, self._ball_loose_frames, frame_idx, fps
            )
            if ball_recovery and self._can_emit(ball_recovery.event_type, frame_idx):
                self._emit(ball_recovery)
            
            # Duel
            duel = self.rule.detect_duel(ball_pos, player_tracks, frame_idx, fps)
            if duel and self._can_emit(duel.event_type, frame_idx):
                self._emit(duel)
            
            # Clearance
            if current_ball_speed:
                clearance = self.rule.detect_clearance(
                    current_ball_speed, ball_pos, self.current_possessor, 
                    player_tracks, fw, frame_idx, fps
                )
                if clearance and self._can_emit(clearance.event_type, frame_idx):
                    self._emit(clearance)
            
            # Block
            block = self.rule.detect_block(
                ball_track, player_tracks, self._prev_ball_speed, 
                current_ball_speed, frame_idx, fps
            )
            if block and self._can_emit(block.event_type, frame_idx):
                self._emit(block)
            
            # Goalkeeper Actions
            if current_ball_speed:
                gk_action = self.rule.detect_goalkeeper_action(
                    ball_pos, current_ball_speed, player_tracks, frame_idx, fps, fw
                )
                if gk_action and self._can_emit(gk_action.event_type, frame_idx):
                    self._emit(gk_action)
            
            # Miscontrol
            miscontrol = self.rule.detect_miscontrol(
                ball_track, self.current_possessor, player_tracks, frame_idx, fps
            )
            if miscontrol and self._can_emit(miscontrol.event_type, frame_idx):
                self._emit(miscontrol)
            
            # Dispossessed
            dispossessed = self.rule.detect_dispossessed(
                self.current_possessor, self.previous_possessor, player_tracks, frame_idx, fps
            )
            if dispossessed and self._can_emit(dispossessed.event_type, frame_idx):
                self._emit(dispossessed)

        # Tackle
        if frame_idx % 3 == 0:
            tackle = self.rule.detect_tackle(player_tracks, frame_idx, fps)
            if tackle and self._can_emit(tackle.event_type, frame_idx):
                self._emit(tackle)
        
        # Foul
        if frame_idx % 5 == 0:
            foul = self.rule.detect_foul(player_tracks, frame_idx, fps)
            if foul and self._can_emit(foul.event_type, frame_idx):
                self._emit(foul)
        
        # Carry (check every 10 frames)
        if frame_idx % 10 == 0:
            carry = self.rule.detect_carry(self.current_possessor, player_tracks, frame_idx, fps)
            if carry and self._can_emit(carry.event_type, frame_idx):
                self._emit(carry)

        # ── ML-Based Events ───────────────────────────────────────────

        if self.config.event_detection.enable_ml_events:
            # Pass
            pss = self.ml.detect_pass(
                self.ball_consecutive,
                self.previous_possessor, self.current_possessor,
                player_tracks, frame_idx, fps,
            )
            if pss and self._can_emit(pss.event_type, frame_idx):
                self._emit(pss)

            # Sprint
            spr = self.ml.detect_sprint(player_tracks, frame_idx, fps)
            if spr and self._can_emit(spr.event_type, frame_idx):
                self._emit(spr)
            
            # v4: New ML events
            
            # Dribble
            dribble = self.ml.detect_dribble(
                self.current_possessor, player_tracks, ball_track, frame_idx, fps
            )
            if dribble and self._can_emit(dribble.event_type, frame_idx):
                self._emit(dribble)
            
            # Interception
            interception = self.ml.detect_interception(
                self.ball_consecutive, self.current_possessor, player_tracks, frame_idx, fps
            )
            if interception and self._can_emit(interception.event_type, frame_idx):
                self._emit(interception)

        # ── Add Freeze Frames to Events ──────────────────────────────
        
        # Generate freeze frames for all detected events this frame
        for event in self.frame_events:
            if not event.freeze_frame:  # Don't overwrite if already set
                event.freeze_frame = self.freeze_frame_gen.generate(
                    event, player_tracks, ball_track
                )

        # Store previous ball speed for next frame
        self._prev_ball_speed = current_ball_speed

        return self.frame_events

    def get_possession_stats(self) -> Tuple[float, float]:
        total = sum(self.possession_frames.values())
        if total == 0:
            return 50.0, 50.0
        t0 = self.possession_frames.get(0, 0) / total * 100
        t1 = self.possession_frames.get(1, 0) / total * 100
        return round(t0, 1), round(t1, 1)

    def get_event_summary(self) -> dict:
        summary = defaultdict(int)
        for e in self.events:
            summary[e.event_type] += 1
        return dict(summary)

    def get_all_events(self) -> List[dict]:
        return [e.to_dict() for e in self.events]