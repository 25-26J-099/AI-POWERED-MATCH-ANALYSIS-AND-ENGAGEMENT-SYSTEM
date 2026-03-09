"""
Simple Ball Tracking Module - Based on Previous Working Implementation

This module replaces the complex Kalman-filter-based ball tracker with a simple,
reliable approach that directly uses YOLO detections with a history buffer.

Philosophy:
- Trust YOLO ball detections (they're already quite good)
- Maintain small history for fallback when ball not detected
- Minimal processing overhead
- Fewer failure modes = more reliability

Based on the proven implementation from component1_tracking.py that worked well.
"""
import numpy as np
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class BallDetection:
    """Single ball detection from YOLO."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    frame_idx: int

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)


@dataclass
class BallTrack:
    """Tracked ball with minimal state information."""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    position_history: deque = field(default_factory=lambda: deque(maxlen=120))
    velocity_history: deque = field(default_factory=lambda: deque(maxlen=30))
    frames_tracked: int = 0
    frames_lost: int = 0
    is_active: bool = True
    last_detection_idx: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    @property
    def current_velocity(self) -> float:
        """Current velocity magnitude in pixels/frame."""
        if len(self.velocity_history) < 1:
            return 0.0
        return self.velocity_history[-1]

    @property
    def avg_velocity(self) -> float:
        """Average velocity over recent history."""
        if not self.velocity_history:
            return 0.0
        return float(np.mean(list(self.velocity_history)))


class SimpleBallTracker:
    """
    Simple ball tracking using direct YOLO detections with history fallback.
    
    Key Design Principles:
    1. Trust YOLO detections (class 32 = sports ball)
    2. Maintain short history buffer for fallback
    3. Use last known position when ball not detected
    4. Minimal processing - let YOLO do the work
    5. Compute velocity from position history
    
    This approach has proven more reliable than complex Kalman filtering
    for amateur football footage with camera shake and motion blur.
    """

    def __init__(self, config, metrics=None):
        """
        Initialize simple ball tracker.
        
        Args:
            config: Configuration object (unused but kept for compatibility)
            metrics: Optional metrics tracker
        """
        # Simple configuration
        self.min_confidence = 0.12  # Class-specific minimum confidence for ball
        self.history_size = 10      # Number of recent positions to remember
        self.max_lost_frames = 30   # Max frames before declaring ball truly lost
        
        # Tracking state
        self.ball_history = deque(maxlen=self.history_size)
        self.track: Optional[BallTrack] = None
        self.next_id = 1
        self.metrics = metrics
        
        # Frame dimensions for boundary checks
        self.frame_width = None
        self.frame_height = None
        
        logger.info(f"[SimpleBallTracker] Initialized (min_conf={self.min_confidence}, "
                   f"history={self.history_size}, max_lost={self.max_lost_frames})")

    def set_metrics_tracker(self, metrics):
        """Set metrics tracker for performance monitoring."""
        self.metrics = metrics
        logger.debug("[SimpleBallTracker] Metrics tracker connected")

    def update(
        self,
        detections: List[BallDetection],
        frame_idx: int,
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[BallTrack]:
        """
        Update ball tracker with new detections.
        
        This is the core tracking logic:
        1. Filter detections by confidence
        2. Select highest confidence detection
        3. Update or create track
        4. Fall back to history if no detection
        
        Args:
            detections: List of ball detection candidates
            frame_idx: Current frame index
            frame_shape: (height, width) of frame
            
        Returns:
            Current ball track or None if lost
        """
        # Set frame dimensions on first call
        if frame_shape is not None and self.frame_width is None:
            self.frame_height, self.frame_width = frame_shape

        # Filter valid detections by confidence
        valid_dets = [d for d in detections if d.confidence >= self.min_confidence]

        # Case 1: Have valid detection(s)
        if valid_dets:
            # Take highest confidence detection (YOLO usually gets it right)
            best_det = max(valid_dets, key=lambda d: d.confidence)
            
            # Update or create track
            if self.track is None or not self.track.is_active:
                # Create new track
                self.track = self._create_track(best_det, frame_idx)
                logger.debug(f"New ball track #{self.track.track_id} created "
                           f"(conf={best_det.confidence:.3f})")
                
                # Track metrics
                if self.metrics:
                    self.metrics.record_ball_track_length(1)
            else:
                # Update existing track
                self._update_track(best_det, frame_idx)
            
            # Add to history buffer
            self.ball_history.append(best_det.center)
            
            return self.track

        # Case 2: No valid detection - use history fallback
        if self.track is not None and self.track.is_active:
            self.track.frames_lost += 1
            
            # Use last known position from history
            if len(self.ball_history) > 0:
                last_pos = self.ball_history[-1]
                
                # Update track with last known position
                w = self.track.bbox[2] - self.track.bbox[0]
                h = self.track.bbox[3] - self.track.bbox[1]
                self.track.bbox = (
                    last_pos[0] - w / 2,
                    last_pos[1] - h / 2,
                    last_pos[0] + w / 2,
                    last_pos[1] + h / 2,
                )
                self.track.position_history.append(last_pos)
                
                # Compute velocity from history
                if len(self.track.position_history) >= 2:
                    prev_pos = self.track.position_history[-2]
                    curr_pos = self.track.position_history[-1]
                    velocity = np.sqrt(
                        (curr_pos[0] - prev_pos[0]) ** 2 +
                        (curr_pos[1] - prev_pos[1]) ** 2
                    )
                    self.track.velocity_history.append(velocity)
                
                logger.debug(f"Ball track #{self.track.track_id} using last position "
                           f"(lost={self.track.frames_lost}/{self.max_lost_frames})")
            
            # Check if track is truly lost
            if self.track.frames_lost > self.max_lost_frames:
                logger.debug(f"Ball track #{self.track.track_id} lost after "
                           f"{self.track.frames_lost} frames")
                
                # Track metrics
                if self.metrics:
                    self.metrics.record_ball_track_length(self.track.frames_tracked)
                
                self.track.is_active = False
                self.track = None
                return None
            
            return self.track

        # Case 3: No track and no detection
        return None

    def _create_track(self, det: BallDetection, frame_idx: int) -> BallTrack:
        """Create new ball track from detection."""
        track = BallTrack(
            track_id=self.next_id,
            bbox=det.bbox,
            confidence=det.confidence,
            frames_tracked=1,
            last_detection_idx=frame_idx,
        )
        track.position_history.append(det.center)
        track.velocity_history.append(0.0)  # Zero initial velocity
        
        self.next_id += 1
        return track

    def _update_track(self, det: BallDetection, frame_idx: int):
        """Update existing track with new detection."""
        self.track.bbox = det.bbox
        self.track.confidence = det.confidence
        self.track.frames_tracked += 1
        self.track.frames_lost = 0
        self.track.last_detection_idx = frame_idx
        
        # Update position history
        self.track.position_history.append(det.center)
        
        # Compute velocity from position change
        if len(self.track.position_history) >= 2:
            prev_pos = self.track.position_history[-2]
            curr_pos = self.track.position_history[-1]
            velocity = np.sqrt(
                (curr_pos[0] - prev_pos[0]) ** 2 +
                (curr_pos[1] - prev_pos[1]) ** 2
            )
            self.track.velocity_history.append(velocity)

    def get_track(self) -> Optional[BallTrack]:
        """Get current active ball track."""
        if self.track is not None and self.track.is_active:
            return self.track
        return None

    def reset(self):
        """Reset tracker state."""
        # Track final track length if metrics available
        if self.metrics and self.track is not None:
            self.metrics.record_ball_track_length(self.track.frames_tracked)
        
        self.track = None
        self.ball_history.clear()
        logger.info("Simple ball tracker reset")


# Maintain compatibility with existing imports
BallTracker = SimpleBallTracker