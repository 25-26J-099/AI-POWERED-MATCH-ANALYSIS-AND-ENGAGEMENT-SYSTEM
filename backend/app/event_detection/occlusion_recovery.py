"""
Occlusion-Aware Track Recovery Module

Targeted fix for ID switching during player crossings/occlusions.
Works alongside ByteTrack to maintain ID consistency when players overlap.

Key Features:
- Predicts reappearance positions for lost tracks
- Motion-based matching (velocity + direction)
- Team-aware (prevents cross-team ID assignment)
- Lightweight (minimal overhead)
- Non-invasive (preserves existing tracking)

Design Philosophy:
ByteTrack handles frame-to-frame tracking
This module handles occlusion recovery
Together = robust ID consistency
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class LostTrackMemory:
    """Memory of a recently lost track for recovery."""
    track_id: int
    last_bbox: Tuple[float, float, float, float]
    last_position: Tuple[float, float]
    velocity: Tuple[float, float]  # px/frame
    team_id: int
    lost_at_frame: int
    position_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    def predict_position(self, frames_elapsed: int) -> Tuple[float, float]:
        """Predict where track should reappear based on velocity."""
        return (
            self.last_position[0] + self.velocity[0] * frames_elapsed,
            self.last_position[1] + self.velocity[1] * frames_elapsed
        )


class OcclusionRecoveryModule:
    """
    Handles ID recovery after occlusions/crossings.
    
    Works in 3 steps:
    1. Remember recently lost tracks with their motion
    2. When new track appears, check if it matches a lost track
    3. If match, restore original ID instead of assigning new
    """
    
    def __init__(self, config):
        self.cfg = config
        
        # Recently lost tracks (potential occlusions)
        self.lost_tracks: Dict[int, LostTrackMemory] = {}
        
        # Configuration
        self.max_recovery_frames = 15  # Max frames to remember lost track
        self.position_threshold = 100.0  # Max distance for recovery match (px)
        self.velocity_threshold = 50.0   # Max velocity change (px/frame)
        self.min_velocity = 2.0          # Min velocity to use motion prediction
        
    def register_lost_track(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        position: Tuple[float, float],
        team_id: int,
        frame_idx: int,
        position_history: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Register a track that was just lost (potential occlusion).
        
        Called when ByteTrack loses a track.
        """
        # Compute velocity from position history
        velocity = (0.0, 0.0)
        if position_history and len(position_history) >= 2:
            # Use last N positions for velocity
            recent = list(position_history)[-min(5, len(position_history)):]
            if len(recent) >= 2:
                dt = len(recent) - 1
                velocity = (
                    (recent[-1][0] - recent[0][0]) / dt,
                    (recent[-1][1] - recent[0][1]) / dt
                )
        
        # Store in memory
        memory = LostTrackMemory(
            track_id=track_id,
            last_bbox=bbox,
            last_position=position,
            velocity=velocity,
            team_id=team_id,
            lost_at_frame=frame_idx,
        )
        
        if position_history:
            memory.position_history = deque(position_history[-10:], maxlen=10)
        
        self.lost_tracks[track_id] = memory
        logger.debug(f"Registered lost track #{track_id} at frame {frame_idx}, "
                    f"velocity=({velocity[0]:.1f}, {velocity[1]:.1f})")
    
    def attempt_recovery(
        self,
        new_bbox: Tuple[float, float, float, float],
        new_position: Tuple[float, float],
        team_id: int,
        frame_idx: int
    ) -> Optional[int]:
        """
        Try to match a new detection to a recently lost track.
        
        Called when ByteTrack creates a NEW track - we check if it's
        actually a recovered old track.
        
        Returns:
            Original track_id if match found, None otherwise
        """
        if not self.lost_tracks:
            return None
        
        best_match_id = None
        best_match_score = float('inf')
        
        for tid, memory in list(self.lost_tracks.items()):
            frames_elapsed = frame_idx - memory.lost_at_frame
            
            # Skip if too old
            if frames_elapsed > self.max_recovery_frames:
                continue
            
            # Team consistency check (hard constraint)
            if team_id >= 0 and memory.team_id >= 0:
                if team_id != memory.team_id:
                    continue
            
            # Predict where track should be
            velocity_mag = np.sqrt(memory.velocity[0]**2 + memory.velocity[1]**2)
            
            if velocity_mag > self.min_velocity:
                # Use motion prediction
                predicted_pos = memory.predict_position(frames_elapsed)
            else:
                # Stationary player - use last known position
                predicted_pos = memory.last_position
            
            # Distance from prediction
            dist = np.sqrt(
                (new_position[0] - predicted_pos[0]) ** 2 +
                (new_position[1] - predicted_pos[1]) ** 2
            )
            
            # Check if within threshold
            if dist > self.position_threshold:
                continue
            
            # Additional check: velocity consistency
            # (new position should roughly follow old velocity direction)
            velocity_score = 0.0
            if velocity_mag > self.min_velocity:
                expected_displacement = (
                    memory.velocity[0] * frames_elapsed,
                    memory.velocity[1] * frames_elapsed
                )
                actual_displacement = (
                    new_position[0] - memory.last_position[0],
                    new_position[1] - memory.last_position[1]
                )
                
                # Dot product of expected vs actual (direction similarity)
                dot = (expected_displacement[0] * actual_displacement[0] +
                       expected_displacement[1] * actual_displacement[1])
                
                if dot < 0:  # Opposite direction - unlikely match
                    velocity_score = 1000.0
                else:
                    # Magnitude difference
                    expected_mag = np.sqrt(
                        expected_displacement[0]**2 + expected_displacement[1]**2
                    )
                    actual_mag = np.sqrt(
                        actual_displacement[0]**2 + actual_displacement[1]**2
                    )
                    velocity_score = abs(actual_mag - expected_mag)
            
            # Combined score (lower is better)
            score = dist + velocity_score * 0.5
            
            if score < best_match_score:
                best_match_score = score
                best_match_id = tid
        
        # If good match found, recover original ID
        if best_match_id is not None and best_match_score < self.position_threshold * 1.5:
            logger.info(f"Recovered track #{best_match_id} after "
                       f"{frame_idx - self.lost_tracks[best_match_id].lost_at_frame} frames "
                       f"(score={best_match_score:.1f})")
            # Remove from lost tracks (recovered)
            del self.lost_tracks[best_match_id]
            return best_match_id
        
        return None
    
    def cleanup_old_memories(self, current_frame: int):
        """Remove lost tracks that are too old to recover."""
        to_remove = [
            tid for tid, memory in self.lost_tracks.items()
            if current_frame - memory.lost_at_frame > self.max_recovery_frames
        ]
        
        for tid in to_remove:
            del self.lost_tracks[tid]
            logger.debug(f"Expired lost track #{tid} from recovery memory")
    
    def get_stats(self) -> Dict:
        """Get statistics about recovery system."""
        return {
            "lost_tracks_in_memory": len(self.lost_tracks),
            "recovery_window_frames": self.max_recovery_frames,
        }
