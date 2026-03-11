"""
Deep Re-ID System - Maximum Accuracy, No Compromises

v2.0: Enhanced with adaptive thresholding and temporal aggregation

Uses:
- Deep appearance features (ResNet-based when available, otherwise rich hand-crafted)
- Strong motion prediction with Kalman filtering
- Conservative ID assignment (high thresholds)
- Adaptive thresholding based on scene context (NEW)
- Enhanced temporal aggregation with confidence voting (NEW)
- Long-term identity memory
- Multi-frame temporal smoothing
- Heavy appearance history buffers

Design: Prioritize CORRECTNESS over speed

Changes in v2.0:
- Added adaptive thresholding for high-velocity and occlusion scenarios
- Integrated temporal aggregator for smoother ID transitions
- Added scene context computation (velocity and occlusion detection)
- Enhanced metrics tracking integration
"""
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# Import temporal aggregator for enhanced ID smoothing
try:
    from app.event_detection.temporal_aggregator import TemporalIDAggregator
    TEMPORAL_AGGREGATOR_AVAILABLE = True
except ImportError:
    TEMPORAL_AGGREGATOR_AVAILABLE = False
    logging.warning("[Re-ID] temporal_aggregator not found, enhanced temporal aggregation disabled")

logger = logging.getLogger(__name__)


class DeepFeatureExtractor:
    """
    Deep feature extraction using ResNet for robust appearance matching.
    Falls back to rich hand-crafted features if GPU unavailable.
    """
    
    def __init__(self):
        self.use_deep = False
        self.device = 'cpu'
        self.model = None
        self.transform = None
        
        # Try to load ResNet
        try:
            # Fix SSL certificate issue on macOS for PyTorch model download
            import ssl
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = models.resnet18(pretrained=True)
            # Remove final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            self.use_deep = True
            logger.info("[Re-ID] Using ResNet18 for deep appearance features")
        except Exception as e:
            logger.info(f"[Re-ID] Deep features unavailable, using hand-crafted: {e}")
    
    def extract(self, frame: np.ndarray, bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract appearance features from player crop."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 - x1 < 10 or y2 - y1 < 20:
            return None
        
        crop = frame[y1:y2, x1:x2]
        
        if self.use_deep:
            return self._extract_deep(crop)
        else:
            return self._extract_handcrafted(crop)
    
    def _extract_deep(self, crop: np.ndarray) -> np.ndarray:
        """Extract deep ResNet features."""
        try:
            img_tensor = self.transform(crop)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()
            
            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
        except Exception as e:
            logger.warning(f"Deep feature extraction failed: {e}")
            return self._extract_handcrafted(crop)
    
    def _extract_handcrafted(self, crop: np.ndarray) -> np.ndarray:
        """Rich hand-crafted features as fallback."""
        crop = cv2.resize(crop, (64, 128))
        
        # 1. Multi-region color histograms
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = crop.shape[:2]
        
        # Jersey region (top 40%)
        jersey = hsv[0:int(h*0.4), :]
        hist_jersey = self._color_hist(jersey, bins=(12, 8, 8))
        
        # Shorts region (middle 40%)
        shorts = hsv[int(h*0.3):int(h*0.7), :]
        hist_shorts = self._color_hist(shorts, bins=(8, 6, 6))
        
        # Full body
        hist_full = self._color_hist(hsv, bins=(8, 6, 6))
        
        # 2. Texture features (HOG)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hog_features = self._hog_features(gray)
        
        # 3. Body shape (contour features)
        shape_features = self._shape_features(gray)
        
        # Combine all features
        features = np.concatenate([
            hist_jersey * 2.0,    # Weight jersey most heavily
            hist_shorts * 1.5,
            hist_full * 1.0,
            hog_features * 0.8,
            shape_features * 0.5
        ])
        
        # Normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _color_hist(self, hsv_img: np.ndarray, bins=(8, 8, 8)) -> np.ndarray:
        """Compute normalized HSV histogram."""
        hist = cv2.calcHist([hsv_img], [0, 1, 2], None, bins,
                           [0, 180, 0, 256, 0, 256])
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-8)
        return hist
    
    def _hog_features(self, gray: np.ndarray) -> np.ndarray:
        """Compute HOG features."""
        # Gradient magnitude and direction
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)
        
        # Histogram of oriented gradients
        hist, _ = np.histogram(angle, bins=9, range=(-np.pi, np.pi),
                              weights=mag)
        hist = hist / (np.sum(hist) + 1e-8)
        return hist
    
    def _shape_features(self, gray: np.ndarray) -> np.ndarray:
        """Body shape features."""
        # Vertical and horizontal projections
        v_proj = np.sum(gray, axis=1)
        h_proj = np.sum(gray, axis=0)
        
        # Normalize
        v_proj = v_proj / (np.sum(v_proj) + 1e-8)
        h_proj = h_proj / (np.sum(h_proj) + 1e-8)
        
        # Downsample for compact representation
        v_proj = cv2.resize(v_proj.reshape(-1, 1), (1, 16)).flatten()
        h_proj = cv2.resize(h_proj.reshape(-1, 1), (1, 8)).flatten()
        
        return np.concatenate([v_proj, h_proj])


class StableIdentity:
    """
    Represents a stable, long-term player identity.
    
    Maintains rich history for robust matching.
    """
    def __init__(self, identity_id: int):
        self.identity_id = identity_id
        self.detection_ids: Set[int] = set()  # All ByteTrack IDs seen
        
        # Appearance features (keep many for robustness)
        self.appearance_history = deque(maxlen=30)  # 30 frames of features
        
        # Position and motion
        self.position_history = deque(maxlen=60)  # 2 seconds at 30fps
        self.velocity_history = deque(maxlen=20)
        
        # Kalman filter for motion prediction
        self.kalman_state = None  # [x, y, vx, vy]
        self.kalman_P = None  # Covariance
        
        # Metadata
        self.team_id = -1
        self.team_votes = []  # History of team assignments
        
        # Tracking state
        self.last_seen_frame = 0
        self.first_seen_frame = 0
        self.total_frames_tracked = 0
        self.frames_since_seen = 0
        
        # Confidence in this identity
        self.confidence = 1.0
        self.match_scores = deque(maxlen=10)  # Recent match quality
    
    def add_observation(self, detection_id: int, features: np.ndarray,
                       position: Tuple[float, float], team_id: int,
                       frame_idx: int):
        """Add new observation to identity."""
        self.detection_ids.add(detection_id)
        
        if features is not None:
            self.appearance_history.append(features)
        
        self.position_history.append(position)
        
        # Update velocity
        if len(self.position_history) >= 2:
            last = self.position_history[-1]
            prev = self.position_history[-2]
            velocity = (last[0] - prev[0], last[1] - prev[1])
            self.velocity_history.append(velocity)
        
        # Update Kalman filter
        self._update_kalman(position)
        
        # Update team
        if team_id >= 0:
            self.team_votes.append(team_id)
            if len(self.team_votes) > 10:
                self.team_votes = self.team_votes[-10:]
            # Majority vote
            if self.team_votes:
                self.team_id = max(set(self.team_votes), key=self.team_votes.count)
        
        self.last_seen_frame = frame_idx
        self.total_frames_tracked += 1
        self.frames_since_seen = 0
    
    def _update_kalman(self, measurement: Tuple[float, float]):
        """Update Kalman filter with new position measurement."""
        if self.kalman_state is None:
            # Initialize
            self.kalman_state = np.array([measurement[0], measurement[1], 0.0, 0.0])
            self.kalman_P = np.eye(4) * 100.0
            return
        
        # Prediction
        F = np.array([[1, 0, 1, 0],
                      [0, 1, 0, 1],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        Q = np.eye(4) * 0.5
        Q[2:, 2:] *= 2.0
        
        self.kalman_state = F @ self.kalman_state
        self.kalman_P = F @ self.kalman_P @ F.T + Q
        
        # Update
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])
        R = np.eye(2) * 4.0
        
        z = np.array(measurement)
        y = z - (H @ self.kalman_state)
        S = H @ self.kalman_P @ H.T + R
        K = self.kalman_P @ H.T @ np.linalg.inv(S)
        
        self.kalman_state = self.kalman_state + K @ y
        self.kalman_P = (np.eye(4) - K @ H) @ self.kalman_P
    
    def predict_position(self, frames_ahead: int = 1) -> Tuple[float, float]:
        """Predict future position."""
        if self.kalman_state is not None:
            # Use Kalman prediction
            pred_state = self.kalman_state.copy()
            for _ in range(frames_ahead):
                F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
                pred_state = F @ pred_state
            return (float(pred_state[0]), float(pred_state[1]))
        elif self.position_history:
            return self.position_history[-1]
        else:
            return (0.0, 0.0)
    
    def get_mean_appearance(self) -> Optional[np.ndarray]:
        """Get mean appearance feature vector."""
        if not self.appearance_history:
            return None
        # Use recent features (last 10)
        recent = list(self.appearance_history)[-10:]
        return np.mean(recent, axis=0)
    
    def compute_appearance_similarity(self, features: np.ndarray) -> float:
        """Compute similarity with this identity's appearance."""
        if not self.appearance_history or features is None:
            return 0.0
        
        # Compare with recent appearances
        recent = list(self.appearance_history)[-10:]
        similarities = []
        for ref_feat in recent:
            sim = np.dot(features, ref_feat) / (
                np.linalg.norm(features) * np.linalg.norm(ref_feat) + 1e-8
            )
            similarities.append(sim)
        
        # Return max similarity (best match)
        return float(np.max(similarities))


class RobustReIDSystem:
    """
    Robust Re-ID system with maximum accuracy focus.
    
    No compromises - uses heavy computation for correctness.
    
    v2.0 Features:
    - Adaptive thresholding based on scene context
    - Enhanced temporal aggregation with confidence voting
    - Metrics tracking integration
    """
    
    def __init__(self, config):
        self.cfg = config
        
        # Deep feature extractor
        self.feature_extractor = DeepFeatureExtractor()
        
        # Identity database
        self.identities: Dict[int, StableIdentity] = {}
        self.next_identity_id = 1
        
        # ByteTrack ID → Stable Identity ID mapping
        self.detection_to_identity: Dict[int, int] = {}
        
        # Base thresholds (may be adapted dynamically)
        self.appearance_threshold = config.reid.appearance_threshold
        self.spatial_threshold = config.reid.spatial_threshold
        self.combined_threshold = config.reid.combined_threshold
        
        # Temporal smoothing (basic)
        self.id_buffer: Dict[int, deque] = {}  # ByteTrack ID → recent stable IDs
        self.buffer_size = 5  # Default 5-frame majority vote
        
        # Memory management
        self.max_identities = 50
        self.max_frames_missing = config.reid.max_lost_frames
        
        # ===== PHASE 1: Adaptive Thresholding =====
        self.use_adaptive = config.reid.use_adaptive_thresholds
        if self.use_adaptive:
            self.velocity_threshold = config.reid.velocity_threshold
            self.high_velocity_appearance_factor = config.reid.high_velocity_appearance_factor
            self.high_velocity_spatial_factor = config.reid.high_velocity_spatial_factor
            self.occlusion_appearance_factor = config.reid.occlusion_appearance_factor
            self.occlusion_spatial_factor = config.reid.occlusion_spatial_factor
            self.occlusion_variance_threshold = config.reid.occlusion_variance_threshold
            logger.info("[Re-ID] Adaptive thresholding: enabled")
        else:
            logger.info("[Re-ID] Adaptive thresholding: disabled")
        
        # ===== PHASE 3: Enhanced Temporal Aggregation =====
        self.use_enhanced_temporal = config.reid.use_enhanced_temporal
        self.temporal_aggregator = None
        
        if self.use_enhanced_temporal:
            if TEMPORAL_AGGREGATOR_AVAILABLE:
                self.temporal_aggregator = TemporalIDAggregator(
                    window_size=config.reid.temporal_window_size,
                    consensus_threshold=config.reid.temporal_consensus_threshold
                )
                logger.info(f"[Re-ID] Enhanced temporal aggregation: enabled "
                          f"(window={config.reid.temporal_window_size}, "
                          f"threshold={config.reid.temporal_consensus_threshold})")
            else:
                logger.warning("[Re-ID] Enhanced temporal aggregation requested but "
                             "temporal_aggregator module not available")
                self.use_enhanced_temporal = False
        else:
            logger.info("[Re-ID] Enhanced temporal aggregation: disabled")
        
        # Metrics tracking (if available)
        self.metrics = None
        self.enable_metrics = config.reid.enable_metrics_tracking
        
        logger.info(f"[Re-ID] Initialized with appearance_threshold={self.appearance_threshold}, "
                   f"spatial_threshold={self.spatial_threshold}")
    
    def set_metrics_tracker(self, metrics):
        """Set metrics tracker for performance monitoring."""
        self.metrics = metrics
        if metrics and self.enable_metrics:
            logger.info("[Re-ID] Metrics tracking enabled")
    
    def _compute_scene_context(self, identity: StableIdentity) -> Tuple[float, float]:
        """
        Compute scene context for adaptive thresholding.
        
        Analyzes identity's recent history to determine:
        - Average velocity (for high-motion detection)
        - Appearance variance (for occlusion detection)
        
        Returns:
            (avg_velocity, occlusion_level)
        """
        # Average velocity over recent history
        avg_velocity = 0.0
        if identity.velocity_history:
            recent_vels = list(identity.velocity_history)[-5:]
            velocities = [np.linalg.norm(v) for v in recent_vels]
            avg_velocity = float(np.mean(velocities))
        
        # Occlusion level (proxy: appearance similarity variance)
        # High variance = inconsistent appearance = likely occlusion/interference
        occlusion_level = 0.0
        if len(identity.appearance_history) > 5:
            recent_features = list(identity.appearance_history)[-5:]
            similarities = []
            for i in range(len(recent_features)-1):
                sim = np.dot(recent_features[i], recent_features[i+1])
                similarities.append(sim)
            
            # High variance = inconsistent appearance = likely occlusion
            if similarities:
                occlusion_level = float(np.std(similarities))
        
        return avg_velocity, occlusion_level
    
    def _get_adaptive_thresholds(self, avg_velocity: float, 
                                occlusion_level: float) -> Tuple[float, float, str]:
        """
        Compute adaptive thresholds based on scene context.
        
        Adjusts matching thresholds dynamically based on:
        - High velocity: Relax appearance, widen spatial (player moving fast)
        - High occlusion: Relax appearance, widen spatial (player partially hidden)
        
        Args:
            avg_velocity: Average velocity magnitude (px/frame)
            occlusion_level: Appearance variance (higher = more occlusion)
            
        Returns:
            (appearance_threshold, spatial_threshold, reason)
            reason: "none", "velocity", "occlusion", or "both"
        """
        app_thresh = self.appearance_threshold
        spa_thresh = self.spatial_threshold
        reason = "none"
        
        high_velocity = avg_velocity > self.velocity_threshold
        high_occlusion = occlusion_level > self.occlusion_variance_threshold
        
        # Apply adaptive adjustments
        if high_velocity and high_occlusion:
            # Both conditions: use most relaxed settings
            app_thresh *= min(self.high_velocity_appearance_factor, 
                             self.occlusion_appearance_factor)
            spa_thresh *= max(self.high_velocity_spatial_factor,
                             self.occlusion_spatial_factor)
            reason = "both"
        elif high_velocity:
            # High velocity: relax appearance, widen spatial
            app_thresh *= self.high_velocity_appearance_factor
            spa_thresh *= self.high_velocity_spatial_factor
            reason = "velocity"
        elif high_occlusion:
            # High occlusion: relax appearance, widen spatial
            app_thresh *= self.occlusion_appearance_factor
            spa_thresh *= self.occlusion_spatial_factor
            reason = "occlusion"
        
        return app_thresh, spa_thresh, reason
    
    def process_frame(self, frame: np.ndarray, tracks: Dict, frame_idx: int) -> Dict:
        """
        Process frame and return tracks with stable IDs.
        
        Args:
            frame: Current video frame
            tracks: ByteTrack output {detection_id: Track}
            frame_idx: Current frame number
            
        Returns:
            Tracks with stable identity IDs
        """
        # Extract features for all detections
        detection_features = {}
        for det_id, track in tracks.items():
            if track.is_ball or track.is_referee:
                continue
            features = self.feature_extractor.extract(frame, track.bbox)
            detection_features[det_id] = features
        
        # Match detections to stable identities
        stable_tracks = {}
        
        for det_id, track in tracks.items():
            if track.is_ball or track.is_referee:
                # Ball and referee keep their IDs
                stable_tracks[det_id] = track
                continue
            
            # Get stable identity for this detection
            identity_id = self._get_stable_identity(
                detection_id=det_id,
                features=detection_features.get(det_id),
                position=track.center,
                team_id=track.team_id,
                frame_idx=frame_idx
            )
            
            # Create track with stable ID
            stable_track = track
            stable_track.track_id = identity_id
            stable_tracks[identity_id] = stable_track
        
        # Update identity states
        self._update_identities(frame_idx)
        
        # Cleanup old identities
        self._cleanup_old_identities(frame_idx)
        
        return stable_tracks
    
    def _get_stable_identity(self, detection_id: int, features: np.ndarray,
                            position: Tuple[float, float], team_id: int,
                            frame_idx: int) -> int:
        """
        Get stable identity ID for a detection.
        
        Uses multi-stage matching with temporal smoothing.
        Enhanced with temporal aggregation for smoother transitions.
        """
        # Stage 1: Check if this detection already has a stable ID
        if detection_id in self.detection_to_identity:
            identity_id = self.detection_to_identity[detection_id]
            if identity_id in self.identities:
                # Update existing identity
                self.identities[identity_id].add_observation(
                    detection_id, features, position, team_id, frame_idx
                )
                self._update_id_buffer(detection_id, identity_id)
                
                # Track metrics
                if self.metrics:
                    self.metrics.record_detection_id(detection_id, identity_id)
                
                return identity_id
        
        # Stage 2: Try to match to existing identity
        matched_id = self._match_to_identity(
            features, position, team_id, frame_idx
        )
        
        if matched_id is not None:
            # Matched to existing identity
            final_id = matched_id
            
            # ===== PHASE 3: Use temporal aggregation if enabled =====
            if self.use_enhanced_temporal and self.temporal_aggregator:
                # Compute match confidence (simplified - could be enhanced)
                # Higher appearance score = higher confidence
                identity = self.identities[matched_id]
                if features is not None:
                    appearance_score = identity.compute_appearance_similarity(features)
                    match_confidence = min(0.95, appearance_score + 0.1)
                else:
                    match_confidence = 0.7
                
                # Add vote to temporal aggregator
                self.temporal_aggregator.add_vote(
                    detection_id=detection_id,
                    identity_id=matched_id,
                    confidence=match_confidence,
                    frame_idx=frame_idx
                )
                
                # Try to get stable ID from aggregator
                stable_id = self.temporal_aggregator.get_stable_id(detection_id)
                if stable_id is not None:
                    final_id = stable_id
                    if stable_id != matched_id:
                        logger.debug(f"Temporal aggregation: {matched_id} -> {stable_id}")
                        
                        # Track metrics
                        if self.metrics:
                            self.metrics.record_consensus_result(achieved=True)
                else:
                    if self.metrics:
                        self.metrics.record_consensus_result(achieved=False)
            
            self.detection_to_identity[detection_id] = final_id
            self.identities[final_id].add_observation(
                detection_id, features, position, team_id, frame_idx
            )
            self._update_id_buffer(detection_id, final_id)
            
            # Track metrics
            if self.metrics:
                self.metrics.record_detection_id(detection_id, final_id)
            
            return final_id
        
        # Stage 3: Create new identity (very conservative)
        if self.metrics:
            self.metrics.record_match_failure()
        
        new_id = self._create_new_identity(
            detection_id, features, position, team_id, frame_idx
        )
        return new_id
    
    def _match_to_identity(self, features: np.ndarray,
                          position: Tuple[float, float],
                          team_id: int, frame_idx: int) -> Optional[int]:
        """
        Match detection to existing identity.
        
        Uses STRICT matching criteria for accuracy.
        Enhanced with adaptive thresholding for challenging scenarios.
        """
        if not self.identities:
            return None
        
        best_match_id = None
        best_score = -1.0
        best_appearance_score = 0.0
        best_spatial_score = 0.0
        best_motion_score = 0.0
        adapt_reason = "none"
        
        for identity_id, identity in self.identities.items():
            # Skip if not seen recently
            if identity.frames_since_seen > self.max_frames_missing:
                continue
            
            # HARD CONSTRAINT: Team consistency
            if team_id >= 0 and identity.team_id >= 0:
                if team_id != identity.team_id:
                    continue  # Skip cross-team matches
            
            # ===== PHASE 1: Compute adaptive thresholds if enabled =====
            if self.use_adaptive:
                avg_vel, occ_level = self._compute_scene_context(identity)
                adaptive_app_thresh, adaptive_spa_thresh, adapt_reason = \
                    self._get_adaptive_thresholds(avg_vel, occ_level)
            else:
                adaptive_app_thresh = self.appearance_threshold
                adaptive_spa_thresh = self.spatial_threshold
                adapt_reason = "none"
            
            # 1. Appearance similarity (PRIMARY)
            appearance_score = 0.0
            if features is not None:
                appearance_score = identity.compute_appearance_similarity(features)
            
            # Must pass adaptive appearance threshold
            if appearance_score < adaptive_app_thresh:
                continue
            
            # 2. Spatial consistency
            predicted_pos = identity.predict_position(
                frames_ahead=identity.frames_since_seen + 1
            )
            dist = np.sqrt(
                (position[0] - predicted_pos[0])**2 +
                (position[1] - predicted_pos[1])**2
            )
            
            # Must pass adaptive spatial threshold
            if dist > adaptive_spa_thresh:
                continue
            
            spatial_score = 1.0 - min(1.0, dist / adaptive_spa_thresh)
            
            # 3. Motion consistency
            motion_score = 0.5  # Neutral if no motion data
            if identity.velocity_history:
                avg_vel = np.mean(list(identity.velocity_history)[-5:], axis=0)
                vel_mag = np.linalg.norm(avg_vel)
                if vel_mag > 1.0:  # Moving
                    # Check if position change aligns with velocity
                    if identity.position_history:
                        last_pos = identity.position_history[-1]
                        actual_disp = (position[0] - last_pos[0],
                                      position[1] - last_pos[1])
                        expected_disp = avg_vel * identity.frames_since_seen
                        
                        disp_diff = np.linalg.norm(
                            np.array(actual_disp) - np.array(expected_disp)
                        )
                        motion_score = 1.0 - min(1.0, disp_diff / 100.0)
            
            # Combined score (appearance is MOST important)
            combined_score = (
                0.60 * appearance_score +  # Appearance dominates
                0.25 * spatial_score +
                0.15 * motion_score
            )
            
            # Confidence bonus (stable identities preferred)
            combined_score *= (0.5 + 0.5 * identity.confidence)
            
            if combined_score > best_score:
                best_score = combined_score
                best_match_id = identity_id
                best_appearance_score = appearance_score
                best_spatial_score = spatial_score
                best_motion_score = motion_score
            
            # Log adaptive threshold usage
            if adapt_reason != "none":
                logger.debug(f"Adaptive thresholds for ID {identity_id}: "
                           f"app={adaptive_app_thresh:.3f}, spa={adaptive_spa_thresh:.1f} "
                           f"(reason: {adapt_reason})")
        
        # Require HIGH match score (conservative)
        if best_match_id is not None and best_score > self.combined_threshold:
            logger.debug(f"Matched to identity {best_match_id}, score={best_score:.3f}")
            
            # Track metrics
            if self.metrics:
                self.metrics.record_match_score(
                    score=best_score,
                    adaptive_used=(adapt_reason != "none"),
                    identity_id=best_match_id,
                    appearance_score=best_appearance_score,
                    spatial_score=best_spatial_score,
                    motion_score=best_motion_score
                )
                
                if adapt_reason != "none":
                    self.metrics.record_adaptive_activation(
                        reason=adapt_reason,
                        appearance_factor=adaptive_app_thresh / self.appearance_threshold,
                        spatial_factor=adaptive_spa_thresh / self.spatial_threshold
                    )
            
            return best_match_id
        
        return None
    
    def _create_new_identity(self, detection_id: int, features: np.ndarray,
                            position: Tuple[float, float], team_id: int,
                            frame_idx: int) -> int:
        """Create new stable identity."""
        new_id = self.next_identity_id
        self.next_identity_id += 1
        
        identity = StableIdentity(new_id)
        identity.add_observation(detection_id, features, position, team_id, frame_idx)
        identity.first_seen_frame = frame_idx
        
        self.identities[new_id] = identity
        self.detection_to_identity[detection_id] = new_id
        self._update_id_buffer(detection_id, new_id)
        
        logger.debug(f"Created new identity {new_id}")
        
        # Track metrics
        if self.metrics:
            self.metrics.record_detection_id(detection_id, new_id)
        
        return new_id
    
    def _update_id_buffer(self, detection_id: int, identity_id: int):
        """Update temporal smoothing buffer."""
        if detection_id not in self.id_buffer:
            self.id_buffer[detection_id] = deque(maxlen=self.buffer_size)
        self.id_buffer[detection_id].append(identity_id)
    
    def _get_smoothed_id(self, detection_id: int) -> int:
        """Get temporally smoothed ID (majority vote)."""
        if detection_id not in self.id_buffer:
            return detection_id
        
        buffer = list(self.id_buffer[detection_id])
        if not buffer:
            return detection_id
        
        # Majority vote
        return max(set(buffer), key=buffer.count)
    
    def _update_identities(self, frame_idx: int):
        """Update frames_since_seen for all identities."""
        for identity in self.identities.values():
            if identity.last_seen_frame < frame_idx:
                identity.frames_since_seen = frame_idx - identity.last_seen_frame
                # Decay confidence when not seen
                identity.confidence *= 0.98
            
            # Track metrics
            if self.metrics and identity.frames_since_seen == 0:
                self.metrics.record_identity_activity(identity.identity_id)
    
    def _cleanup_old_identities(self, frame_idx: int):
        """Remove very old identities."""
        to_remove = []
        for identity_id, identity in self.identities.items():
            if identity.frames_since_seen > self.max_frames_missing * 2:
                to_remove.append(identity_id)
        
        for identity_id in to_remove:
            del self.identities[identity_id]
            # Clean up mappings
            to_remove_det = [det_id for det_id, id_id in self.detection_to_identity.items()
                            if id_id == identity_id]
            for det_id in to_remove_det:
                del self.detection_to_identity[det_id]
        
        # Limit total identities
        if len(self.identities) > self.max_identities:
            # Remove least confident
            sorted_ids = sorted(self.identities.items(),
                              key=lambda x: x[1].confidence)
            for identity_id, _ in sorted_ids[:len(self.identities) - self.max_identities]:
                del self.identities[identity_id]
        
        # ===== PHASE 3: Cleanup temporal aggregator =====
        if self.use_enhanced_temporal and self.temporal_aggregator:
            active_det_ids = list(self.detection_to_identity.keys())
            self.temporal_aggregator.cleanup(active_det_ids, frame_idx)