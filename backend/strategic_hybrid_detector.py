"""
Strategic Hybrid Event Detector with Event Routing

This module implements an intelligent event routing system that directs each
event type to the most appropriate detector (ML, Rule-based, or Hybrid).

Key improvements over naive hybrid:
1. Clear event ownership (no redundancy)
2. Optimized computation (ML only for complex events)
3. Better accuracy (each detector does what it's best at)
4. Configurable routing per event type
5. Hybrid fusion for events that benefit from both

CORRECTED VERSION: Uses correct imports from existing codebase.
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DetectorType(Enum):
    """Which detector should handle this event."""
    RULE_ONLY = "rule_only"     # Rule-based detection only
    ML_ONLY = "ml_only"         # ML detection only
    HYBRID = "hybrid"           # Both detectors, fused together


class EventType(Enum):
    """All supported event types."""
    # Existing events
    POSSESSION_CHANGE = "possession_change"
    PASS_ATTEMPT = "pass"
    SHOT = "shot"
    TACKLE = "tackle"
    OUT_OF_BOUNDS = "out_of_bounds"
    SPRINT = "sprint"
    
    # Additional events
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
    """Ensure plain Python float for serialization."""
    if v is None:
        return 0.0
    return float(v)


# Import existing classes from your codebase
from modules.event_detector import GameEvent, RuleBasedDetector, MLEventDetector, FreezeFrameGenerator


class EventRouter:
    """
    Routes events to the appropriate detector based on configuration.
    
    This is the core intelligence of the strategic hybrid system.
    """
    
    def __init__(self, routing_config: Dict[str, List[str]]):
        """
        Initialize event router.
        
        Args:
            routing_config: Dict with keys "ml_only", "rule_only", "hybrid"
                          and values as lists of event type names
        """
        self.routing_config = routing_config
        
        # Build reverse lookup: event_type -> detector_type
        self.event_to_detector = {}
        
        for event in routing_config.get("ml_only", []):
            self.event_to_detector[event] = DetectorType.ML_ONLY
        
        for event in routing_config.get("rule_only", []):
            self.event_to_detector[event] = DetectorType.RULE_ONLY
        
        for event in routing_config.get("hybrid", []):
            self.event_to_detector[event] = DetectorType.HYBRID
        
        # Log routing configuration
        logger.info("[Event Router] Initialized with routing:")
        logger.info(f"  ML-only events: {routing_config.get('ml_only', [])}")
        logger.info(f"  Rule-only events: {routing_config.get('rule_only', [])}")
        logger.info(f"  Hybrid events: {routing_config.get('hybrid', [])}")
    
    def should_use_ml(self, event_type: str) -> bool:
        """Check if ML detector should run for this event."""
        detector = self.event_to_detector.get(event_type, DetectorType.RULE_ONLY)
        return detector in [DetectorType.ML_ONLY, DetectorType.HYBRID]
    
    def should_use_rule(self, event_type: str) -> bool:
        """Check if rule detector should run for this event."""
        detector = self.event_to_detector.get(event_type, DetectorType.RULE_ONLY)
        return detector in [DetectorType.RULE_ONLY, DetectorType.HYBRID]
    
    def get_detector_type(self, event_type: str) -> DetectorType:
        """Get the detector type for an event."""
        return self.event_to_detector.get(event_type, DetectorType.RULE_ONLY)


class FusionLayer:
    """
    Combines ML and rule-based detections for hybrid events.
    
    For hybrid events, we use:
    - ML for event classification and confidence
    - Rules for spatial context, timing, and actor identification
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fuse_events(
        self, 
        ml_event: Optional[GameEvent],
        rule_event: Optional[GameEvent]
    ) -> Optional[GameEvent]:
        """
        Combine ML and rule-based events into a single hybrid event.
        
        Strategy:
        - If both detected: Use ML confidence + rule spatial data
        - If only ML: Use ML but mark lower confidence
        - If only rule: Use rule but mark as fallback
        """
        if ml_event and rule_event:
            # Best case: Both detected, combine strengths
            hybrid_event = GameEvent(
                event_type=ml_event.event_type,
                frame_idx=rule_event.frame_idx,  # Rule has frame-exact timing
                timestamp=rule_event.timestamp,
                confidence=ml_event.confidence * 0.7 + 0.3,  # Boost confidence
                position=rule_event.position,  # Rule has precise position
                player_id=rule_event.player_id,  # Rule identifies actor
                team_id=rule_event.team_id,
                details=f"Hybrid: ML({ml_event.confidence:.2f}) + Rule",
                source="hybrid",
                freeze_frame=rule_event.freeze_frame
            )
            return hybrid_event
        
        elif ml_event:
            # ML only: Keep but reduce confidence slightly
            ml_event.confidence *= 0.85  # Slight penalty for no rule confirmation
            ml_event.source = "hybrid_ml"
            ml_event.details = f"Hybrid: ML only ({ml_event.confidence:.2f})"
            return ml_event
        
        elif rule_event:
            # Rule only: Keep but mark as fallback
            rule_event.source = "hybrid_rule"
            rule_event.details = f"Hybrid: Rule fallback"
            return rule_event
        
        return None


class StrategicHybridEventDetector:
    """
    Strategic hybrid event detector with intelligent event routing.
    
    This is a drop-in replacement for the current HybridEventDetector,
    but with optimized routing and no redundancy.
    """
    
    def __init__(self, config):
        """
        Initialize strategic hybrid detector.
        
        Args:
            config: PipelineConfig with event_routing configuration
        """
        self.config = config
        
        # Initialize existing detectors (using your existing classes)
        self.rule = RuleBasedDetector(config)
        self.ml = MLEventDetector(config)
        self.freeze_frame_gen = FreezeFrameGenerator()
        
        # Initialize routing system
        self.router = EventRouter(config.event_routing.routing)
        self.fusion = FusionLayer()
        
        # Event storage
        self.events = []
        self.frame_events = []
        
        # Cooldown tracking (per event type)
        self.last_event_frame = defaultdict(lambda: -999)
        
        # Event cooldowns (from config or defaults)
        self.event_cooldowns = getattr(config.event_detection, 'event_cooldowns', {
            "out_of_bounds": 100,
            "shot": 60,
            "tackle": 50,
            "possession_change": 25,
            "pass": 25,
            "sprint": 100,
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
        })
        
        # Stats
        self.possession_frames = defaultdict(int)
        self.current_possessor = None
        self.previous_possessor = None
        
        # Ball tracking for complex detections
        self.ball_consecutive = deque(maxlen=30)
        self._prev_ball_speed = None
        self._ball_loose_frames = 0
        
        logger.info("[Strategic Hybrid] Initialized with intelligent event routing")
    
    def _can_emit(self, event_type: str, frame_idx: int) -> bool:
        """Check if enough time has passed to emit this event type."""
        cooldown = self.event_cooldowns.get(event_type, 30)
        last_frame = self.last_event_frame[event_type]
        return (frame_idx - last_frame) > cooldown
    
    def _emit(self, event: GameEvent):
        """Add event to both frame events and global events list."""
        self.frame_events.append(event)
        self.events.append(event)
        self.last_event_frame[event.event_type] = event.frame_idx
    
    def _update_ball_history(self, ball_track, frame_idx):
        """Update ball position history for velocity calculations."""
        if ball_track is not None and ball_track.frames_lost < 5:
            self.ball_consecutive.append((
                frame_idx,
                _f(ball_track.center[0]),
                _f(ball_track.center[1])
            ))
        else:
            self.ball_consecutive.clear()
    
    def _ball_speed_consecutive(self) -> Optional[float]:
        """Calculate ball speed from consecutive frames."""
        if len(self.ball_consecutive) < 5:
            return None
        recent = list(self.ball_consecutive)[-5:]
        dx = recent[-1][1] - recent[0][1]
        dy = recent[-1][2] - recent[0][2]
        frames = recent[-1][0] - recent[0][0]
        if frames == 0:
            return 0.0
        return ((dx*dx + dy*dy)**0.5) / frames
    
    def _ball_traveled_distance(self) -> float:
        """Calculate total distance ball traveled in consecutive frames."""
        if len(self.ball_consecutive) < 2:
            return 0.0
        start = self.ball_consecutive[0]
        end = self.ball_consecutive[-1]
        return ((end[1] - start[1])**2 + (end[2] - start[2])**2)**0.5
    
    def _detect_shot(self, fw, frame_idx, fps) -> Optional[GameEvent]:
        """Detect shot using ball speed and direction."""
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
                event_type="shot",
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                confidence=0.65,
                position=(_f(bx), _f(by)),
                details=f"Shot (speed={speed:.1f}px/f)",
                source="rule",
            )
        return None
    
    def process_frame(
        self,
        frame,
        player_tracks,
        ball_track,
        frame_idx,
        fps=25.0
    ) -> List[GameEvent]:
        """
        Process frame using strategic hybrid detection.
        
        This is the main entry point that routes events to appropriate detectors.
        """
        self.frame_events = []
        fh, fw = frame.shape[:2]
        
        # Update ball history
        self._update_ball_history(ball_track, frame_idx)
        
        ball_pos = None
        if ball_track is not None and ball_track.frames_lost < 5:
            ball_pos = (_f(ball_track.center[0]), _f(ball_track.center[1]))
        
        # Calculate ball metrics
        current_ball_speed = self._ball_speed_consecutive()
        ball_traveled_dist = self._ball_traveled_distance()
        
        # ═══════════════════════════════════════════════════════════
        # STRATEGIC EVENT DETECTION WITH ROUTING
        # ═══════════════════════════════════════════════════════════
        
        if ball_pos is not None:
            # ─── RULE-ONLY EVENTS ───────────────────────────────────
            
            # Possession (CRITICAL - always rule-based)
            if self.router.should_use_rule("possession_change"):
                poss_ev = self.rule.detect_possession(
                    ball_pos, player_tracks, frame_idx, fps
                )
                if poss_ev and self._can_emit(poss_ev.event_type, frame_idx):
                    self._emit(poss_ev)
            
            # Track possessor
            self.previous_possessor = self.current_possessor
            if self.rule.possession_player is not None:
                self.current_possessor = self.rule.possession_player
                team = self.rule.possession_team
                if team >= 0:
                    self.possession_frames[team] += 1
            
            # Out of Bounds (CRITICAL - always rule-based)
            if self.router.should_use_rule("out_of_bounds"):
                oob = self.rule.detect_out_of_bounds(
                    ball_pos, fw, fh, frame_idx, fps
                )
                if oob and self._can_emit(oob.event_type, frame_idx):
                    self._emit(oob)
            
            # Shot (rule-based for accuracy)
            if self.router.should_use_rule("shot"):
                shot = self._detect_shot(fw, frame_idx, fps)
                if shot and self._can_emit(shot.event_type, frame_idx):
                    self._emit(shot)
            
            # Ball Receipt (geometric - rule-based)
            if self.router.should_use_rule("ball_receipt"):
                ball_receipt = self.rule.detect_ball_receipt(
                    self.current_possessor, self.previous_possessor,
                    ball_traveled_dist, frame_idx, fps, player_tracks
                )
                if ball_receipt and self._can_emit(ball_receipt.event_type, frame_idx):
                    self._emit(ball_receipt)
            
            # Pressure (geometric - rule-based)
            if self.router.should_use_rule("pressure"):
                pressure = self.rule.detect_pressure(
                    self.current_possessor, player_tracks, frame_idx, fps
                )
                if pressure and self._can_emit(pressure.event_type, frame_idx):
                    self._emit(pressure)
            
            # Ball Recovery (rule-based)
            if self.router.should_use_rule("ball_recovery"):
                ball_recovery = self.rule.detect_ball_recovery(
                    self.current_possessor, self.previous_possessor,
                    player_tracks, self._ball_loose_frames, frame_idx, fps
                )
                if ball_recovery and self._can_emit(ball_recovery.event_type, frame_idx):
                    self._emit(ball_recovery)
            
            # Clearance (rule-based)
            if self.router.should_use_rule("clearance") and current_ball_speed:
                clearance = self.rule.detect_clearance(
                    current_ball_speed, ball_pos, self.current_possessor,
                    player_tracks, fw, frame_idx, fps
                )
                if clearance and self._can_emit(clearance.event_type, frame_idx):
                    self._emit(clearance)
            
            # Block (rule-based)
            if self.router.should_use_rule("block"):
                block = self.rule.detect_block(
                    ball_track, player_tracks, self._prev_ball_speed,
                    current_ball_speed, frame_idx, fps
                )
                if block and self._can_emit(block.event_type, frame_idx):
                    self._emit(block)
            
            # Goalkeeper Actions (rule-based)
            if self.router.should_use_rule("goalkeeper_save") and current_ball_speed:
                gk_action = self.rule.detect_goalkeeper_action(
                    ball_pos, current_ball_speed, player_tracks, frame_idx, fps, fw
                )
                if gk_action and self._can_emit(gk_action.event_type, frame_idx):
                    self._emit(gk_action)
            
            # Miscontrol (rule-based)
            if self.router.should_use_rule("miscontrol"):
                miscontrol = self.rule.detect_miscontrol(
                    ball_track, self.current_possessor, player_tracks, frame_idx, fps
                )
                if miscontrol and self._can_emit(miscontrol.event_type, frame_idx):
                    self._emit(miscontrol)
            
            # Dispossessed (rule-based)
            if self.router.should_use_rule("dispossessed"):
                dispossessed = self.rule.detect_dispossessed(
                    self.current_possessor, self.previous_possessor,
                    player_tracks, frame_idx, fps
                )
                if dispossessed and self._can_emit(dispossessed.event_type, frame_idx):
                    self._emit(dispossessed)
        
        # Duel (rule-based, checked less frequently)
        if ball_pos and frame_idx % 3 == 0:
            if self.router.should_use_rule("duel"):
                duel = self.rule.detect_duel(
                    ball_pos, player_tracks, frame_idx, fps
                )
                if duel and self._can_emit(duel.event_type, frame_idx):
                    self._emit(duel)
        
        # Tackle (check routing - could be rule, ML, or hybrid)
        if frame_idx % 3 == 0:
            if self.router.should_use_rule("tackle"):
                tackle = self.rule.detect_tackle(player_tracks, frame_idx, fps)
                if tackle and self._can_emit(tackle.event_type, frame_idx):
                    self._emit(tackle)
        
        # Foul (check routing - could be rule, ML, or hybrid)
        if frame_idx % 5 == 0:
            if self.router.should_use_rule("foul"):
                foul = self.rule.detect_foul(player_tracks, frame_idx, fps)
                if foul and self._can_emit(foul.event_type, frame_idx):
                    self._emit(foul)
        
        # Carry (check routing - could be rule, ML, or hybrid)
        if frame_idx % 10 == 0:
            if self.router.should_use_rule("carry"):
                carry = self.rule.detect_carry(
                    self.current_possessor, player_tracks, frame_idx, fps
                )
                if carry and self._can_emit(carry.event_type, frame_idx):
                    self._emit(carry)
        
        # ─── ML-ONLY EVENTS ─────────────────────────────────────
        
        if self.config.event_detection.enable_ml_events:
            # Pass (ML is better at temporal patterns)
            if self.router.should_use_ml("pass"):
                pss = self.ml.detect_pass(
                    self.ball_consecutive,
                    self.previous_possessor, self.current_possessor,
                    player_tracks, frame_idx, fps
                )
                if pss and self._can_emit(pss.event_type, frame_idx):
                    self._emit(pss)
            
            # Sprint (ML is better at sustained movement)
            if self.router.should_use_ml("sprint"):
                spr = self.ml.detect_sprint(player_tracks, frame_idx, fps)
                if spr and self._can_emit(spr.event_type, frame_idx):
                    self._emit(spr)
            
            # Dribble (ML is better at complex patterns)
            if self.router.should_use_ml("dribble"):
                dribble = self.ml.detect_dribble(
                    self.current_possessor, player_tracks, ball_track, frame_idx, fps
                )
                if dribble and self._can_emit(dribble.event_type, frame_idx):
                    self._emit(dribble)
            
            # Interception (ML is better at trajectory analysis)
            if self.router.should_use_ml("interception"):
                interception = self.ml.detect_interception(
                    self.ball_consecutive, self.current_possessor,
                    player_tracks, frame_idx, fps
                )
                if interception and self._can_emit(interception.event_type, frame_idx):
                    self._emit(interception)
        
        # ─── HYBRID EVENTS (Future enhancement) ─────────────────
        # For events routed to "hybrid", we would:
        # 1. Get ML detection
        # 2. Get rule detection  
        # 3. Fuse with FusionLayer
        # This can be added later when needed
        
        # ─── Add Freeze Frames ──────────────────────────────────
        for event in self.frame_events:
            if not event.freeze_frame:
                event.freeze_frame = self.freeze_frame_gen.generate(
                    event, player_tracks, ball_track
                )
        
        # Update previous ball speed
        self._prev_ball_speed = current_ball_speed
        
        return self.frame_events
    
    def get_possession_stats(self) -> Tuple[float, float]:
        """Get possession percentage for each team."""
        total = sum(self.possession_frames.values())
        if total == 0:
            return 50.0, 50.0
        t0 = self.possession_frames.get(0, 0) / total * 100
        t1 = self.possession_frames.get(1, 0) / total * 100
        return round(t0, 1), round(t1, 1)
    
    def get_event_summary(self) -> dict:
        """Get count of each event type."""
        summary = defaultdict(int)
        for e in self.events:
            summary[e.event_type] += 1
        return dict(summary)
    
    def get_all_events(self) -> List[dict]:
        """Get all events as dictionaries."""
        return [e.to_dict() for e in self.events]