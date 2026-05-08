"""
Module 2: Lightweight Player and Ball Tracking.

Core tracking system using:
- YOLOv8n (nano) for object detection — optimized for speed on low-resource hardware
- ByteTrack algorithm for multi-object tracking — robust against occlusions

The ByteTrack two-stage matching retains low-confidence detections,
which is critical for handling motion blur and poor lighting in amateur footage.

v4: Robust ball tracking using Kalman filter and ByteTrack principles.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

# Import the new robust ball tracker
from app.event_detection.ball_tracker import BallTracker, BallDetection as BallDet, BallTrack as BallTrk


@dataclass
class Detection:
    """Single object detection result."""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str = ""


@dataclass
class Track:
    """Tracked object with persistent identity."""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = ""
    team_id: int = -1
    jersey_number: Optional[str] = None
    jersey_confidence: float = 0.0
    is_ball: bool = False
    is_referee: bool = False
    # History for velocity/trajectory
    position_history: list = field(default_factory=list)
    frames_tracked: int = 0
    frames_lost: int = 0
    is_active: bool = True

    @property
    def center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    @property
    def bottom_center(self) -> Tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            self.bbox[3],
        )

    @property
    def velocity(self) -> float:
        if len(self.position_history) < 2:
            return 0.0
        p1 = self.position_history[-2]
        p2 = self.position_history[-1]
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


class YOLODetector:
    """
    Object detector for football scene understanding.
    
    Supports multiple backends:
    1. Ultralytics YOLOv8 (preferred - if installed)
    2. OpenCV DNN with YOLO ONNX model (fallback)
    3. OpenCV HOG + contour-based detection (minimal fallback)
    """

    # COCO class names for reference
    COCO_NAMES = {0: "person", 32: "sports ball"}

    def __init__(self, config):
        self.cfg = config.detection
        self.model = None
        self._backend = None  # "ultralytics", "opencv_dnn", or "opencv_hog"
        self._initialized = False
        self._hog = None
        self._bg_subtractor = None
        self._yolo_device = "cpu"
        self._yolo_half = False

    def initialize(self):
        """Load detection model with automatic backend selection."""
        if self._initialized:
            return

        # Try ultralytics first
        try:
            from ultralytics import YOLO
            device = self._resolve_device()
            self._yolo_device = device  # store — must be passed at inference time too
            self._yolo_half = device == "cuda"

            logger.info(f"Loading YOLO model: {self.cfg.model_name} on {device}")
            self.model = YOLO(self.cfg.model_name)
            # NOTE: model.to(device) moves weights but ultralytics resolves the
            # inference device from call kwargs — NOT from weight location.
            # We must pass device= explicitly in every self.model() call.
            self.model.to(device)
            try:
                self.model.fuse()
            except Exception:
                pass
            if device == "cuda":
                try:
                    import torch
                    torch.backends.cudnn.benchmark = True
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
            self._backend = "ultralytics"
            self._initialized = True
            logger.info(
                "[OK] Using Ultralytics YOLOv8 backend on device: %s half=%s batch_size=%s",
                device,
                self._yolo_half,
                getattr(self.cfg, "batch_size", 1),
            )
            return
        except (ImportError, Exception) as e:
            logger.info(f"Ultralytics not available ({e}), trying fallbacks...")

        # Try OpenCV DNN with ONNX model
        import os
        onnx_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", "yolov8n.onnx"
        )
        if os.path.exists(onnx_path):
            try:
                self.model = cv2.dnn.readNetFromONNX(onnx_path)
                self._backend = "opencv_dnn"
                self._initialized = True
                logger.info("[OK] Using OpenCV DNN backend with ONNX model")
                return
            except Exception as e:
                logger.info(f"OpenCV DNN failed: {e}")

        # Fallback: OpenCV HOG person detector + color-based ball detection
        logger.info("Using OpenCV HOG + contour fallback detector")
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=50, detectShadows=False
        )
        self._backend = "opencv_hog"
        self._initialized = True
        logger.info("[OK] Using OpenCV HOG + contour-based fallback detector")

    def _resolve_device(self) -> str:
        preferred = str(getattr(self.cfg, "device", "auto") or "auto").lower()
        if preferred not in {"auto", "cpu", "cuda"}:
            preferred = "auto"

        if preferred == "cpu":
            logger.info("[Device] Forced to CPU by config.")
            return "cpu"

        try:
            import torch
        except ImportError:
            logger.warning("[Device] torch not importable. Using CPU.")
            return "cpu"

        if not torch.cuda.is_available():
            logger.warning(
                "[Device] CUDA not available inside this container. Using CPU. "
                "Check that nvidia-container-toolkit is installed and 'runtime: nvidia' "
                "is set in docker-compose.yml for the backend service."
            )
            return "cpu"

        # Trust torch.cuda.is_available() — skip the arch-list check.
        # PyTorch 2.x includes sm_89 (NVIDIA L4 / RTX 40-series) in cu124 wheels.
        # The old arch-list guard was overly conservative and silently caused YOLO
        # to run on CPU on machines with sm_89 GPUs in some environment configurations.
        gpu_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        logger.info(
            "[Device] Using CUDA — GPU: %s (sm_%s%s)",
            gpu_name, capability[0], capability[1],
        )
        return "cuda"

    def _ultralytics_kwargs(self, batch_size: int = 1) -> dict:
        min_conf = min(self.cfg.player_confidence, self.cfg.ball_confidence)
        return {
            "conf": min_conf,
            "iou": self.cfg.iou_threshold,
            "imgsz": self.cfg.input_size,
            "device": self._yolo_device,
            "half": self._yolo_half,
            "classes": [self.cfg.person_class_id, self.cfg.ball_class_id],
            "batch": max(1, int(batch_size)),
            "verbose": False,
        }

    def _detections_from_ultralytics_result(self, result) -> List[Detection]:
        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].detach().cpu().numpy()

            if cls_id == self.cfg.person_class_id:
                if conf < self.cfg.player_confidence:
                    continue
                class_name = "player"
            elif cls_id == self.cfg.ball_class_id:
                if conf < self.cfg.ball_confidence:
                    continue
                class_name = "ball"
            else:
                continue

            detections.append(Detection(
                bbox=(float(xyxy[0]), float(xyxy[1]),
                      float(xyxy[2]), float(xyxy[3])),
                confidence=conf,
                class_id=cls_id,
                class_name=class_name,
            ))
        return detections

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        """
        Detection using Ultralytics YOLOv8 with class-specific confidence filtering.
        
        Uses lowest threshold for YOLO inference, then applies class-specific
        thresholds for optimal per-class performance.
        """
        # device= MUST be passed here — ultralytics resolves inference device from
        # call kwargs, not from where model weights are. Omitting it causes CPU
        # inference even when the model is loaded on GPU.
        results = self.model(
            frame,
            **self._ultralytics_kwargs(batch_size=1),
        )[0]
        return self._detections_from_ultralytics_result(results)

    def _detect_ultralytics_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        if not frames:
            return []
        results = self.model(
            frames,
            **self._ultralytics_kwargs(batch_size=min(len(frames), int(getattr(self.cfg, "batch_size", 1) or 1))),
        )
        return [self._detections_from_ultralytics_result(result) for result in results]

    def _detect_opencv_dnn(self, frame: np.ndarray) -> List[Detection]:
        """Detection using OpenCV DNN backend."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (640, 640), swapRB=True, crop=False
        )
        self.model.setInput(blob)
        outputs = self.model.forward(self.model.getUnconnectedOutLayersNames())

        detections = []
        # Parse YOLO output format
        for output in outputs:
            for detection in output[0]:
                scores = detection[4:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])

                if confidence < self.cfg.confidence_threshold:
                    continue
                if class_id not in (self.cfg.person_class_id, self.cfg.ball_class_id):
                    continue

                cx, cy, bw, bh = detection[:4]
                x1 = (cx - bw/2) * w / 640
                y1 = (cy - bh/2) * h / 640
                x2 = (cx + bw/2) * w / 640
                y2 = (cy + bh/2) * h / 640

                class_name = "player" if class_id == self.cfg.person_class_id else "ball"
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                ))
        return detections

    def _detect_opencv_hog(self, frame: np.ndarray) -> List[Detection]:
        """
        Fallback detection using OpenCV HOG (person) + contour analysis (ball).
        Works without any neural network model files.
        """
        detections = []
        h, w = frame.shape[:2]

        # --- Person detection via HOG ---
        # Resize for HOG efficiency
        scale = min(1.0, 640 / max(w, h))
        if scale < 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale)
        else:
            small = frame

        boxes, weights = self._hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05,
        )

        for (x, y, bw, bh), weight in zip(boxes, weights):
            conf = min(1.0, float(weight) / 2.0)
            if conf < self.cfg.confidence_threshold:
                continue

            # Scale back to original resolution
            x1 = int(x / scale)
            y1 = int(y / scale)
            x2 = int((x + bw) / scale)
            y2 = int((y + bh) / scale)

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                class_id=0,
                class_name="player",
            ))

        # --- Ball detection via color + contour ---
        # Look for small, round, white/bright objects
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # White ball mask
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))
        # Bright colored ball mask
        bright_mask = cv2.inRange(hsv, (0, 80, 180), (180, 255, 255))
        ball_mask = cv2.bitwise_or(white_mask, bright_mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Ball should be small relative to frame (5-500 px area typical)
            if area < 5 or area > min(w, h) * 2:
                continue

            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity < 0.4:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            # Aspect ratio check
            aspect = bw / max(bh, 1)
            if aspect < 0.5 or aspect > 2.0:
                continue

            detections.append(Detection(
                bbox=(x, y, x + bw, y + bh),
                confidence=0.5 * circularity,
                class_id=32,
                class_name="ball",
            ))

        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame using the best available backend."""
        if not self._initialized:
            self.initialize()

        if self._backend == "ultralytics":
            return self._detect_ultralytics(frame)
        elif self._backend == "opencv_dnn":
            return self._detect_opencv_dnn(frame)
        else:
            return self._detect_opencv_hog(frame)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames."""
        if not frames:
            return []
        if not self._initialized:
            self.initialize()

        if self._backend == "ultralytics":
            return self._detect_ultralytics_batch(frames)
        return [self.detect(f) for f in frames]


class ByteTrackTracker:
    """
    ByteTrack-inspired multi-object tracker.

    Key innovation: two-stage matching that retains ALL detection boxes,
    including low-confidence ones. This is critical for low-quality video
    where motion blur and poor lighting produce valid but low-confidence detections.
    """

    def __init__(self, config):
        self.cfg = config.tracking
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_count = 0
        self._use_supervision = False

    def initialize(self):
        """Try to use supervision's ByteTrack, fallback to custom."""
        try:
            import supervision as sv
            try:
                self.byte_tracker = sv.ByteTrack(
                    track_activation_threshold=self.cfg.track_high_thresh,
                    lost_track_buffer=self.cfg.track_buffer,
                    minimum_matching_threshold=self.cfg.match_thresh,
                    frame_rate=30,
                    minimum_consecutive_frames=2,
                )
            except TypeError:
                self.byte_tracker = sv.ByteTrack(
                    track_activation_threshold=self.cfg.track_high_thresh,
                    lost_track_buffer=self.cfg.track_buffer,
                    minimum_matching_threshold=self.cfg.match_thresh,
                    frame_rate=30,
                )
            self._use_supervision = True
            logger.info("[OK] Using supervision ByteTrack")
        except Exception as exc:
            logger.info("[OK] Using custom ByteTrack implementation (supervision unavailable: %s)", exc)
            self._use_supervision = False

    def _iou(self, bbox1, bbox2) -> float:
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        a2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0
    
    def _compute_motion_cost(self, det_bbox, track) -> float:
        """
        Compute motion-based matching cost.
        
        Uses velocity prediction to determine if detection matches track trajectory.
        Critical for maintaining IDs during crossings/occlusions.
        """
        if not hasattr(track, 'position_history') or len(track.position_history) < 2:
            return 0.0  # No motion history, rely on IoU only
        
        # Current detection center
        det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
        
        # Predict where track should be based on velocity
        recent = list(track.position_history)[-min(5, len(track.position_history)):]
        if len(recent) >= 2:
            dt = len(recent) - 1
            velocity = (
                (recent[-1][0] - recent[0][0]) / dt,
                (recent[-1][1] - recent[0][1]) / dt
            )
            
            # Predicted next position
            predicted = (
                recent[-1][0] + velocity[0],
                recent[-1][1] + velocity[1]
            )
            
            # Distance from prediction
            dist = np.sqrt(
                (det_center[0] - predicted[0]) ** 2 +
                (det_center[1] - predicted[1]) ** 2
            )
            
            # Normalize to 0-1 range (assume max plausible movement is 150 pixels/frame)
            motion_cost = min(1.0, dist / 150.0)
            return motion_cost
        
        return 0.0
    
    def _compute_appearance_cost(self, det_bbox, track, frame) -> float:
        """
        Compute appearance-based matching cost using jersey colors.
        
        Critical for distinguishing between similar players during crossings.
        """
        if frame is None or not hasattr(track, 'team_id'):
            return 0.0
        
        try:
            # Extract jersey region (upper 40% of bbox)
            x1, y1, x2, y2 = [int(v) for v in det_bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 - x1 < 5 or y2 - y1 < 10:
                return 0.0
            
            bbox_h = y2 - y1
            jersey_y2 = y1 + int(bbox_h * 0.4)
            jersey_crop = frame[y1:jersey_y2, x1:x2]
            
            if jersey_crop.size == 0:
                return 0.0
            
            # Extract color histogram
            hsv = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Compare with track's stored appearance if available
            if hasattr(track, 'appearance_features') and track.appearance_features:
                ref_hist = track.appearance_features[-1]  # Most recent
                similarity = cv2.compareHist(hist, ref_hist, cv2.HISTCMP_CORREL)
                appearance_cost = 1.0 - max(0, similarity)  # Convert correlation to cost
                
                # Store current appearance for future comparisons
                track.appearance_features.append(hist)
                if len(track.appearance_features) > 5:
                    track.appearance_features = track.appearance_features[-5:]
                
                return appearance_cost
            else:
                # First time - store appearance
                track.appearance_features = [hist]
                return 0.0
                
        except Exception:
            return 0.0
        
        return 0.0

    def _hungarian_match(
        self,
        detections: List[Detection],
        tracks: Dict[int, Track],
        iou_threshold: float,
        frame: np.ndarray = None,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to tracks using HYBRID cost function.
        
        Combines:
        - IoU (spatial overlap)
        - Motion prediction (velocity-based)
        - Appearance (jersey color similarity)
        - Team consistency (hard constraint)
        
        This prevents ID switching during crossings/occlusions.
        """
        if not detections or not tracks:
            return [], list(range(len(detections))), list(tracks.keys())

        track_ids = list(tracks.keys())
        cost_matrix = np.zeros((len(detections), len(track_ids)))

        for d_idx, det in enumerate(detections):
            for t_idx, tid in enumerate(track_ids):
                track = tracks[tid]
                
                # 1. IoU cost (spatial overlap) - weight: 0.4
                iou = self._iou(det.bbox, track.bbox)
                iou_cost = 1 - iou
                
                # 2. Motion cost (velocity prediction) - weight: 0.35
                motion_cost = self._compute_motion_cost(det.bbox, track)
                
                # 3. Appearance cost (jersey color) - weight: 0.25
                appearance_cost = 0.0
                if frame is not None:
                    appearance_cost = self._compute_appearance_cost(det.bbox, track, frame)
                
                # 4. Team consistency (hard constraint)
                team_penalty = 0.0
                if hasattr(track, 'team_id') and track.team_id >= 0:
                    if hasattr(det, 'team_id') and det.team_id >= 0:
                        if track.team_id != det.team_id:
                            team_penalty = 10.0  # Prohibitive cost for team mismatch
                
                # Combined hybrid cost
                combined_cost = (
                    0.40 * iou_cost +
                    0.35 * motion_cost +
                    0.25 * appearance_cost +
                    team_penalty
                )
                
                cost_matrix[d_idx, t_idx] = combined_cost

        # Simple greedy matching (Hungarian approximation)
        matches = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))

        # Sort by minimum cost
        if cost_matrix.size > 0:
            while unmatched_dets and unmatched_tracks:
                min_cost = float('inf')
                best_d, best_t = -1, -1
                for d in unmatched_dets:
                    for t in unmatched_tracks:
                        if cost_matrix[d, t] < min_cost:
                            min_cost = cost_matrix[d, t]
                            best_d, best_t = d, t

                # Adjusted threshold for hybrid cost (higher threshold since it's multi-cue)
                if min_cost > 0.7:  # Was: (1 - iou_threshold), now adjusted for hybrid
                    break

                matches.append((best_d, track_ids[best_t]))
                unmatched_dets.remove(best_d)
                unmatched_tracks.remove(best_t)

        unmatched_track_ids = [track_ids[t] for t in unmatched_tracks]
        return matches, unmatched_dets, unmatched_track_ids

    def update_custom(self, detections: List[Detection], frame: np.ndarray = None) -> Dict[int, Track]:
        """
        Custom ByteTrack-style two-stage matching with HYBRID cost.

        Stage 1: Match high-confidence detections to existing tracks
        Stage 2: Match low-confidence detections to remaining unmatched tracks
        
        Uses IoU + Motion + Appearance for robust ID consistency during crossings.
        """
        self.frame_count += 1

        # Split detections by confidence
        high_conf = [d for d in detections if d.confidence >= self.cfg.track_high_thresh]
        low_conf = [d for d in detections
                     if self.cfg.track_low_thresh <= d.confidence < self.cfg.track_high_thresh]

        active_tracks = {
            tid: t for tid, t in self.tracks.items() if t.is_active
        }

        # Stage 1: Match high-confidence detections (with frame for appearance)
        matches1, unmatched_dets1, unmatched_tracks1 = self._hungarian_match(
            high_conf, active_tracks, self.cfg.match_thresh, frame=frame
        )

        # Update matched tracks
        matched_track_ids = set()
        for d_idx, tid in matches1:
            det = high_conf[d_idx]
            self.tracks[tid].bbox = det.bbox
            self.tracks[tid].confidence = det.confidence
            self.tracks[tid].frames_tracked += 1
            self.tracks[tid].frames_lost = 0
            self.tracks[tid].position_history.append(
                ((det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2)
            )
            # Keep history manageable
            if len(self.tracks[tid].position_history) > 120:
                self.tracks[tid].position_history = self.tracks[tid].position_history[-60:]
            matched_track_ids.add(tid)

        # Stage 2: Match low-confidence detections to remaining tracks (with frame)
        remaining_tracks = {
            tid: t for tid, t in active_tracks.items()
            if tid in unmatched_tracks1
        }

        matches2, unmatched_dets2, unmatched_tracks2 = self._hungarian_match(
            low_conf, remaining_tracks, self.cfg.match_thresh * 0.8, frame=frame
        )

        for d_idx, tid in matches2:
            det = low_conf[d_idx]
            self.tracks[tid].bbox = det.bbox
            self.tracks[tid].confidence = det.confidence
            self.tracks[tid].frames_tracked += 1
            self.tracks[tid].frames_lost = 0
            self.tracks[tid].position_history.append(
                ((det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2)
            )
            matched_track_ids.add(tid)

        # Create new tracks for unmatched high-confidence detections
        for d_idx in unmatched_dets1:
            det = high_conf[d_idx]
            if det.confidence >= self.cfg.new_track_thresh:
                new_track = Track(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    is_ball=(det.class_name == "ball"),
                    position_history=[
                        ((det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2)
                    ],
                    frames_tracked=1,
                )
                # Initialize appearance features
                new_track.appearance_features = []
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Mark unmatched tracks as lost
        for tid in unmatched_tracks2:
            if tid not in matched_track_ids:
                self.tracks[tid].frames_lost += 1
                if self.tracks[tid].frames_lost > self.cfg.track_buffer:
                    self.tracks[tid].is_active = False

        # Return only active tracks
        return {
            tid: t for tid, t in self.tracks.items()
            if t.is_active and t.frames_tracked >= self.cfg.min_track_length
        }

    def update_supervision(self, detections: List[Detection], frame: np.ndarray) -> Dict[int, Track]:
        """Update using supervision ByteTrack."""
        import supervision as sv

        if not detections:
            return {}

        xyxy = np.array([d.bbox for d in detections])
        confidence = np.array([d.confidence for d in detections])
        class_id = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked = self.byte_tracker.update_with_detections(sv_detections)

        result = {}
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                bbox = tuple(tracked.xyxy[i])
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
                cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                cls_name = "ball" if cls_id == 32 else "player"

                if tid not in self.tracks:
                    self.tracks[tid] = Track(
                        track_id=tid,
                        bbox=bbox,
                        confidence=conf,
                        class_id=cls_id,
                        class_name=cls_name,
                        is_ball=(cls_name == "ball"),
                        position_history=[],
                        frames_tracked=0,
                    )

                track = self.tracks[tid]
                track.bbox = bbox
                track.confidence = conf
                track.frames_tracked += 1
                track.frames_lost = 0
                track.is_active = True
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                track.position_history.append(center)
                if len(track.position_history) > 120:
                    track.position_history = track.position_history[-60:]

                result[tid] = track

        return result

    def update(self, detections: List[Detection], frame: np.ndarray = None) -> Dict[int, Track]:
        """Main update method - routes to appropriate tracker."""
        if self._use_supervision and frame is not None:
            return self.update_supervision(detections, frame)
        # Pass frame to custom tracker for hybrid matching
        return self.update_custom(detections, frame=frame)

    def get_all_tracks(self) -> Dict[int, Track]:
        """Return all tracks including inactive ones (for Re-ID)."""
        return dict(self.tracks)

    def get_lost_tracks(self, max_lost_frames: int = 90) -> Dict[int, Track]:
        """Return recently lost tracks for Re-ID matching."""
        return {
            tid: t for tid, t in self.tracks.items()
            if not t.is_active and t.frames_lost <= max_lost_frames
        }


class PlayerBallTracker:
    """
    Combined player and ball tracking system.
    Orchestrates YOLOv8 detection + ByteTrack tracking.
    
    v4: Uses robust Kalman-filter-based ball tracker for temporal consistency.
    """

    def __init__(self, config, detector: Optional[YOLODetector] = None):
        self.config = config
        self.detector = detector or YOLODetector(config)
        self.tracker = ByteTrackTracker(config)
        self.ball_tracker = BallTracker(config)  # New robust ball tracker
        self.player_tracks: Dict[int, Track] = {}
        self._ball_track_wrapper: Optional[Track] = None

    def initialize(self):
        """Initialize detection and tracking models."""
        self.detector.initialize()
        self.tracker.initialize()
        logger.info("[OK] Robust ball tracker initialized (Kalman filter + ByteTrack)")

    def _convert_ball_track(self, ball_track: Optional[BallTrk], frame_idx: int) -> Optional[Track]:
        """Convert BallTrack to Track format for compatibility with existing pipeline."""
        if ball_track is None or not ball_track.is_active:
            return None
        
        # Create or update wrapper Track
        if self._ball_track_wrapper is None:
            self._ball_track_wrapper = Track(
                track_id=-1,  # Ball always has ID -1
                bbox=ball_track.bbox,
                confidence=ball_track.confidence,
                class_id=32,  # sports ball class
                class_name="ball",
                is_ball=True,
                position_history=list(ball_track.position_history),
                frames_tracked=ball_track.frames_tracked,
                frames_lost=ball_track.frames_lost,
            )
        else:
            # Update existing wrapper
            self._ball_track_wrapper.bbox = ball_track.bbox
            self._ball_track_wrapper.confidence = ball_track.confidence
            self._ball_track_wrapper.frames_tracked = ball_track.frames_tracked
            self._ball_track_wrapper.frames_lost = ball_track.frames_lost
            self._ball_track_wrapper.position_history = list(ball_track.position_history)
        
        return self._ball_track_wrapper

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Tuple[Dict[int, Track], Optional[Track]]:
        """
        Process a single frame through detection + tracking pipeline.

        Returns:
            (player_tracks, ball_track): Dict of player tracks and the ball track
        """
        # Detect objects
        detections = self.detector.detect(frame)
        return self.process_detections(frame, detections, frame_idx)

    def process_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_idx: int = 0,
    ) -> Tuple[Dict[int, Track], Optional[Track]]:
        """Update player and ball trackers from precomputed detections."""
        # Separate player and ball detections
        player_dets = [d for d in detections if d.class_name == "player"]
        ball_dets = [d for d in detections if d.class_name == "ball"]

        # Track players using ByteTrack
        active_tracks = self.tracker.update(player_dets, frame)

        # Separate player tracks (exclude any ball detections)
        self.player_tracks = {
            tid: t for tid, t in active_tracks.items()
            if not t.is_ball
        }

        # Track ball using robust Kalman-based tracker
        ball_detections = [
            BallDet(bbox=d.bbox, confidence=d.confidence, frame_idx=frame_idx)
            for d in ball_dets
        ]
        
        ball_track = self.ball_tracker.update(
            ball_detections,
            frame_idx,
            frame_shape=frame.shape[:2]
        )
        
        # Convert to Track format for compatibility
        ball_track_wrapped = self._convert_ball_track(ball_track, frame_idx)

        return self.player_tracks, ball_track_wrapped

    def get_lost_tracks(self) -> Dict[int, Track]:
        """Get recently lost tracks for Re-ID module."""
        return self.tracker.get_lost_tracks(
            max_lost_frames=self.config.reid.max_lost_frames
        )
