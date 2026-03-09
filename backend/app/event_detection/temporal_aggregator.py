"""
Temporal ID Aggregation with Confidence-Weighted Voting

Provides smooth ID transitions by aggregating votes over multiple frames
with confidence weighting. This replaces simple majority voting with a more
sophisticated approach that reduces ID flickers.

Key Features:
- Confidence-weighted consensus voting
- Smooth ID transitions (no sudden jumps)
- Configurable window size and consensus threshold
- Automatic cleanup of inactive detections
"""
from collections import deque, defaultdict
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalIDAggregator:
    """
    Aggregates identity assignments over time with confidence weighting.
    
    This class maintains a voting history for each detection and uses
    confidence-weighted consensus to determine stable identity assignments.
    """
    
    def __init__(self, window_size: int = 10, consensus_threshold: float = 0.65):
        """
        Initialize temporal aggregator.
        
        Args:
            window_size: Number of frames to consider for voting (default: 10)
            consensus_threshold: Minimum confidence ratio required for consensus (default: 0.65)
        """
        self.window_size = window_size
        self.consensus_threshold = consensus_threshold
        
        # Track votes for each detection
        # {detection_id: deque of (identity_id, confidence, frame_idx)}
        self.vote_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Last stable ID for each detection (for fallback)
        self.last_stable_id: Dict[int, int] = {}
        
        # Statistics
        self.consensus_achieved = 0
        self.consensus_failed = 0
    
    def add_vote(self, detection_id: int, identity_id: int, 
                 confidence: float, frame_idx: int):
        """
        Add a vote for identity assignment.
        
        Args:
            detection_id: ByteTrack detection ID
            identity_id: Proposed stable identity ID
            confidence: Confidence score for this assignment (0-1)
            frame_idx: Current frame index
        """
        self.vote_history[detection_id].append(
            (identity_id, confidence, frame_idx)
        )
    
    def get_stable_id(self, detection_id: int) -> Optional[int]:
        """
        Get stable identity ID using confidence-weighted consensus.
        
        This method computes a weighted vote across the temporal window:
        - Each vote is weighted by its confidence
        - The identity with the highest weighted vote wins
        - Consensus is required (best_vote / total_votes >= threshold)
        
        Args:
            detection_id: Detection to get stable ID for
            
        Returns:
            Stable identity ID, or None if no strong consensus
        """
        if detection_id not in self.vote_history:
            return None
        
        votes = list(self.vote_history[detection_id])
        if len(votes) < 3:  # Need at least 3 votes for meaningful consensus
            return None
        
        # Count weighted votes for each identity
        id_votes: Dict[int, float] = defaultdict(float)
        total_confidence = 0.0
        
        for identity_id, confidence, _ in votes:
            id_votes[identity_id] += confidence
            total_confidence += confidence
        
        if total_confidence == 0:
            return None
        
        # Find ID with highest weighted vote
        best_id = max(id_votes.keys(), key=lambda k: id_votes[k])
        best_confidence_ratio = id_votes[best_id] / total_confidence
        
        # Check if consensus threshold met
        if best_confidence_ratio >= self.consensus_threshold:
            self.last_stable_id[detection_id] = best_id
            self.consensus_achieved += 1
            logger.debug(f"Consensus achieved for det {detection_id}: ID {best_id} "
                        f"(confidence: {best_confidence_ratio:.2f})")
            return best_id
        
        # No strong consensus, return last stable ID if exists
        self.consensus_failed += 1
        if detection_id in self.last_stable_id:
            logger.debug(f"No consensus for det {detection_id}, using last stable ID "
                        f"{self.last_stable_id[detection_id]}")
            return self.last_stable_id[detection_id]
        
        return None
    
    def get_vote_distribution(self, detection_id: int) -> Dict[int, float]:
        """
        Get the vote distribution for a detection (for debugging/visualization).
        
        Args:
            detection_id: Detection to analyze
            
        Returns:
            Dictionary mapping identity_id to normalized vote weight
        """
        if detection_id not in self.vote_history:
            return {}
        
        votes = list(self.vote_history[detection_id])
        if not votes:
            return {}
        
        id_votes: Dict[int, float] = defaultdict(float)
        total_confidence = 0.0
        
        for identity_id, confidence, _ in votes:
            id_votes[identity_id] += confidence
            total_confidence += confidence
        
        if total_confidence == 0:
            return {}
        
        # Normalize
        return {id_: vote / total_confidence for id_, vote in id_votes.items()}
    
    def cleanup(self, active_detection_ids: List[int], current_frame: int):
        """
        Remove old detections that are no longer active.
        
        This prevents memory accumulation for detections that have ended.
        
        Args:
            active_detection_ids: List of currently active detection IDs
            current_frame: Current frame index
        """
        to_remove = []
        
        for det_id, votes in self.vote_history.items():
            if det_id not in active_detection_ids:
                # Check if last vote is old (more than 2x window size)
                if votes and current_frame - votes[-1][2] > self.window_size * 2:
                    to_remove.append(det_id)
        
        for det_id in to_remove:
            del self.vote_history[det_id]
            if det_id in self.last_stable_id:
                del self.last_stable_id[det_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} inactive detections from aggregator")
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get aggregator statistics.
        
        Returns:
            Dictionary with consensus statistics
        """
        return {
            'consensus_achieved': self.consensus_achieved,
            'consensus_failed': self.consensus_failed,
            'active_detections': len(self.vote_history),
            'stable_ids_tracked': len(self.last_stable_id)
        }
    
    def reset(self):
        """Reset all vote history (for new video or testing)."""
        self.vote_history.clear()
        self.last_stable_id.clear()
        self.consensus_achieved = 0
        self.consensus_failed = 0