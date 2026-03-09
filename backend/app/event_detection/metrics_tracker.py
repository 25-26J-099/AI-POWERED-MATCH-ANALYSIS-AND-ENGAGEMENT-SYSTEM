"""
Performance Metrics Tracking for Re-ID and Tracking Quality

Tracks various metrics to quantify system performance and improvements:
- ID switches and stability
- Match scores and confidence
- Adaptive threshold activations
- Ball tracking quality

This module provides objective measurements to validate improvements.
"""
from collections import defaultdict
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Track Re-ID and tracking performance metrics."""
    
    def __init__(self):
        # ID stability metrics
        self.id_switches = defaultdict(list)  # {frame: [switch_events]}
        self.id_stability = defaultdict(int)  # {identity_id: frames_active}
        self.detection_to_id_history = defaultdict(list)  # {detection_id: [identity_ids]}
        
        # Matching metrics
        self.match_scores = []
        self.match_failures = 0
        
        # Adaptive thresholding metrics
        self.adaptive_activations = {
            'velocity': 0,
            'occlusion': 0,
            'both': 0
        }
        self.threshold_adjustments = []
        
        # Ball tracking metrics
        self.ball_detections = 0
        self.ball_track_continuity = []  # List of track lengths
        self.ball_size_rejections = 0
        
        # Frame counter
        self.total_frames = 0
        
        # Temporal aggregation metrics
        self.consensus_achieved = 0
        self.consensus_failed = 0
    
    def record_id_switch(self, frame_idx: int, detection_id: int, 
                        old_id: int, new_id: int, reason: str = ""):
        """
        Record an ID switch event.
        
        Args:
            frame_idx: Frame where switch occurred
            detection_id: ByteTrack detection ID
            old_id: Previous identity ID
            new_id: New identity ID
            reason: Optional reason for switch
        """
        self.id_switches[frame_idx].append({
            'detection': detection_id,
            'old_id': old_id,
            'new_id': new_id,
            'reason': reason
        })
        logger.debug(f"ID switch at frame {frame_idx}: det={detection_id}, "
                    f"{old_id} -> {new_id} ({reason})")
    
    def record_detection_id(self, detection_id: int, identity_id: int):
        """Track identity assignments over time for each detection."""
        self.detection_to_id_history[detection_id].append(identity_id)
    
    def record_match_score(self, score: float, adaptive_used: bool = False,
                          identity_id: Optional[int] = None,
                          appearance_score: float = 0.0,
                          spatial_score: float = 0.0,
                          motion_score: float = 0.0):
        """
        Record matching score.
        
        Args:
            score: Combined match score
            adaptive_used: Whether adaptive thresholding was used
            identity_id: Identity that was matched
            appearance_score: Appearance component
            spatial_score: Spatial component
            motion_score: Motion component
        """
        self.match_scores.append({
            'score': score,
            'adaptive': adaptive_used,
            'identity_id': identity_id,
            'appearance': appearance_score,
            'spatial': spatial_score,
            'motion': motion_score
        })
    
    def record_match_failure(self):
        """Record that no match was found."""
        self.match_failures += 1
    
    def record_adaptive_activation(self, reason: str, 
                                  appearance_factor: float = 1.0,
                                  spatial_factor: float = 1.0):
        """
        Record when adaptive thresholding was triggered.
        
        Args:
            reason: 'velocity', 'occlusion', or 'both'
            appearance_factor: Adjustment factor for appearance threshold
            spatial_factor: Adjustment factor for spatial threshold
        """
        if reason in self.adaptive_activations:
            self.adaptive_activations[reason] += 1
        
        self.threshold_adjustments.append({
            'reason': reason,
            'appearance_factor': appearance_factor,
            'spatial_factor': spatial_factor
        })
    
    def record_identity_activity(self, identity_id: int):
        """Record that an identity was active this frame."""
        self.id_stability[identity_id] += 1
    
    def record_ball_detection(self, accepted: bool, size_rejected: bool = False):
        """
        Record ball detection result.
        
        Args:
            accepted: Whether detection was accepted
            size_rejected: Whether rejected due to size validation
        """
        self.ball_detections += 1
        if size_rejected:
            self.ball_size_rejections += 1
    
    def record_ball_track_length(self, length: int):
        """Record length of a completed ball track."""
        self.ball_track_continuity.append(length)
    
    def record_consensus_result(self, achieved: bool):
        """Record temporal aggregator consensus result."""
        if achieved:
            self.consensus_achieved += 1
        else:
            self.consensus_failed += 1
    
    def update_frame_count(self):
        """Increment total frame counter."""
        self.total_frames += 1
    
    def compute_id_switch_rate(self) -> float:
        """
        Compute ID switches per detection per frame.
        
        Returns:
            Average ID switch rate
        """
        if not self.detection_to_id_history:
            return 0.0
        
        total_switches = 0
        for det_id, id_history in self.detection_to_id_history.items():
            if len(id_history) > 1:
                # Count transitions
                switches = sum(1 for i in range(len(id_history)-1) 
                             if id_history[i] != id_history[i+1])
                total_switches += switches
        
        return total_switches / len(self.detection_to_id_history)
    
    def get_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        total_switches = sum(len(switches) for switches in self.id_switches.values())
        
        avg_match_score = (
            np.mean([m['score'] for m in self.match_scores]) 
            if self.match_scores else 0
        )
        
        avg_appearance_score = (
            np.mean([m['appearance'] for m in self.match_scores if m['appearance'] > 0])
            if any(m['appearance'] > 0 for m in self.match_scores) else 0
        )
        
        avg_spatial_score = (
            np.mean([m['spatial'] for m in self.match_scores if m['spatial'] > 0])
            if any(m['spatial'] > 0 for m in self.match_scores) else 0
        )
        
        # Switches per minute (assuming 30 fps)
        switches_per_minute = (
            (total_switches / self.total_frames) * 30 * 60 
            if self.total_frames > 0 else 0
        )
        
        # Ball tracking metrics
        ball_acceptance_rate = (
            (self.ball_detections - self.ball_size_rejections) / max(1, self.ball_detections)
        )
        
        avg_ball_track_length = (
            np.mean(self.ball_track_continuity) 
            if self.ball_track_continuity else 0
        )
        
        # Adaptive thresholding effectiveness
        total_adaptive = sum(self.adaptive_activations.values())
        adaptive_rate = total_adaptive / max(1, self.total_frames)
        
        # Consensus metrics
        consensus_rate = (
            self.consensus_achieved / max(1, self.consensus_achieved + self.consensus_failed)
        )
        
        return {
            # Frame counts
            'total_frames': self.total_frames,
            
            # ID stability
            'total_id_switches': total_switches,
            'switches_per_frame': total_switches / max(1, self.total_frames),
            'switches_per_minute': switches_per_minute,
            'id_switch_rate': self.compute_id_switch_rate(),
            'unique_identities': len(self.id_stability),
            'avg_frames_per_identity': (
                np.mean(list(self.id_stability.values())) 
                if self.id_stability else 0
            ),
            
            # Matching quality
            'avg_match_score': avg_match_score,
            'avg_appearance_score': avg_appearance_score,
            'avg_spatial_score': avg_spatial_score,
            'match_failures': self.match_failures,
            'total_matches': len(self.match_scores),
            'match_success_rate': (
                len(self.match_scores) / max(1, len(self.match_scores) + self.match_failures)
            ),
            
            # Adaptive thresholding
            'adaptive_velocity_activations': self.adaptive_activations['velocity'],
            'adaptive_occlusion_activations': self.adaptive_activations['occlusion'],
            'adaptive_both_activations': self.adaptive_activations['both'],
            'total_adaptive_activations': total_adaptive,
            'adaptive_activation_rate': adaptive_rate,
            
            # Temporal aggregation
            'consensus_achieved': self.consensus_achieved,
            'consensus_failed': self.consensus_failed,
            'consensus_rate': consensus_rate,
            
            # Ball tracking
            'ball_detections': self.ball_detections,
            'ball_size_rejections': self.ball_size_rejections,
            'ball_acceptance_rate': ball_acceptance_rate,
            'avg_ball_track_length': avg_ball_track_length,
            'total_ball_tracks': len(self.ball_track_continuity),
        }
    
    def save(self, path: str):
        """
        Save metrics to JSON file.
        
        Args:
            path: Output JSON path
        """
        summary = self.get_summary()
        
        output = {
            'summary': summary,
            'id_switches': {
                str(k): v for k, v in self.id_switches.items()
            },
            'threshold_adjustments': self.threshold_adjustments[:100],  # Limit size
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Print summary to console
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE METRICS SUMMARY")
        logger.info("="*60)
        
        logger.info("\n🎯 ID Stability:")
        logger.info(f"  Total ID switches: {summary['total_id_switches']}")
        logger.info(f"  Switches per minute: {summary['switches_per_minute']:.2f}")
        logger.info(f"  ID switch rate: {summary['id_switch_rate']:.4f}")
        logger.info(f"  Unique identities: {summary['unique_identities']}")
        
        logger.info("\n📊 Matching Quality:")
        logger.info(f"  Avg match score: {summary['avg_match_score']:.3f}")
        logger.info(f"  Avg appearance score: {summary['avg_appearance_score']:.3f}")
        logger.info(f"  Match success rate: {summary['match_success_rate']:.2%}")
        
        logger.info("\n⚙️  Adaptive Thresholding:")
        logger.info(f"  Velocity activations: {summary['adaptive_velocity_activations']}")
        logger.info(f"  Occlusion activations: {summary['adaptive_occlusion_activations']}")
        logger.info(f"  Activation rate: {summary['adaptive_activation_rate']:.4f}")
        
        logger.info("\n🔄 Temporal Aggregation:")
        logger.info(f"  Consensus achieved: {summary['consensus_achieved']}")
        logger.info(f"  Consensus rate: {summary['consensus_rate']:.2%}")
        
        logger.info("\n⚽ Ball Tracking:")
        logger.info(f"  Ball detections: {summary['ball_detections']}")
        logger.info(f"  Size rejections: {summary['ball_size_rejections']}")
        logger.info(f"  Acceptance rate: {summary['ball_acceptance_rate']:.2%}")
        logger.info(f"  Avg track length: {summary['avg_ball_track_length']:.1f} frames")
        
        logger.info("\n" + "="*60)
        logger.info(f"Metrics saved to: {path}")
        logger.info("="*60 + "\n")
    
    def print_summary(self):
        """Print summary to console without saving."""
        summary = self.get_summary()
        
        print("\n=== Performance Metrics ===")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
