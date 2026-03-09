"""
ML Event Detector Integration Module

Integrates trained event detection model into existing pipeline.
Runs alongside rule-based detector in a hybrid system.

IMPORTANT: This version is designed to work with the user's EventDetectionModel
from their training code (ResNet18 + LSTM architecture).
"""
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import logging

# Import the user's model architecture
from app.models.event_detection_model import EventDetectionModel

logger = logging.getLogger(__name__)


class MLEventDetector:
    """
    ML-based event detector using trained temporal CNN.
    Maintains frame buffer for temporal window processing.
    """
    
    def __init__(self, weights_path: str, config, device='auto'):
        """
        Initialize ML event detector.
        
        Args:
            weights_path: Path to event_detector_weights.pth
            config: Pipeline configuration
            device: Device for inference ('cuda', 'cpu', or 'auto')
        """
        self.config = config
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self._load_model(weights_path)
        
        # Frame buffer for temporal window
        self.frame_buffer = deque(maxlen=self.temporal_window)
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Inference configuration
        self.confidence_threshold = 0.6
        self.inference_interval = 5  # Run every N frames
        self.frame_count = 0
        
        logger.info(f"[ML Detector] Loaded on {self.device}")
        logger.info(f"[ML Detector] Temporal window: {self.temporal_window} frames")
        logger.info(f"[ML Detector] Event classes: {len(self.class_names)}")
    
    def _load_model(self, weights_path: str):
        """
        Load model from exported weights file.
        
        Args:
            weights_path: Path to event_detector_weights.pth
        """
        try:
            # Load checkpoint
            self.checkpoint = torch.load(weights_path, map_location=self.device)
            
            # Extract metadata
            self.class_names = self.checkpoint['class_names']
            self.event_mapping = self.checkpoint['event_mapping']
            
            # Extract model configuration
            model_config = self.checkpoint.get('model_config', {})
            
            # Get parameters (with defaults matching user's training code)
            num_classes = model_config.get('num_classes', 17)
            sequence_length = model_config.get('temporal_window', 16)
            hidden_dim = model_config.get('hidden_size', 512)
            lstm_layers = model_config.get('num_lstm_layers', 2)
            dropout = model_config.get('dropout', 0.5)
            
            # Store temporal window
            self.temporal_window = sequence_length
            
            # Create model using USER'S architecture
            self.model = EventDetectionModel(
                num_classes=num_classes,
                sequence_length=sequence_length,
                pretrained=False,  # Don't download ImageNet weights
                hidden_dim=hidden_dim,
                lstm_layers=lstm_layers,
                dropout=dropout
            )
            
            # Load trained weights
            self.model.load_state_dict(self.checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"[ML Detector] Model loaded successfully")
            logger.info(f"  Architecture: ResNet18 + LSTM")
            logger.info(f"  Hidden dim: {hidden_dim}")
            logger.info(f"  LSTM layers: {lstm_layers}")
            
            if 'best_val_acc' in self.checkpoint:
                logger.info(f"  Best accuracy: {self.checkpoint['best_val_acc']:.2f}%")
            
        except Exception as e:
            logger.error(f"[ML Detector] Failed to load model: {e}")
            raise
    
    def update_buffer(self, frame):
        """
        Add frame to buffer.
        
        Args:
            frame: RGB frame (H, W, 3) as numpy array
        """
        # Preprocess and add to buffer
        try:
            processed = self.transform(frame)
            self.frame_buffer.append(processed)
            self.frame_count += 1
        except Exception as e:
            logger.error(f"[ML Detector] Error processing frame: {e}")
    
    def predict(self):
        """
        Run inference on current buffer.
        
        Returns:
            predictions: Dict with class, confidence, and mapped event
                        Returns None if buffer not full or not inference frame
        """
        # Check if buffer is full
        if len(self.frame_buffer) < self.temporal_window:
            return None
        
        # Only run inference every N frames
        if self.frame_count % self.inference_interval != 0:
            return None
        
        try:
            # Stack frames into tensor
            frames = torch.stack(list(self.frame_buffer))  # [T, C, H, W]
            frames = frames.unsqueeze(0)  # [1, T, C, H, W]
            frames = frames.to(self.device)
            
            # Run inference
            with torch.no_grad():
                logits = self.model(frames)  # [1, num_classes]
                probs = torch.softmax(logits, dim=1)
            
            # Get top prediction
            confidence, pred_idx = torch.max(probs, dim=1)
            confidence = confidence.item()
            pred_idx = pred_idx.item()
            pred_class = self.class_names[pred_idx]
            
            # Map to system event type
            mapped_event = self.event_mapping.get(pred_class, pred_class.lower())
            
            return {
                'class': pred_class,
                'confidence': confidence,
                'mapped_event': mapped_event,
                'all_probs': probs.cpu().numpy()[0]
            }
        
        except Exception as e:
            logger.error(f"[ML Detector] Inference error: {e}")
            return None
    
    def reset(self):
        """Clear buffer and reset state."""
        self.frame_buffer.clear()
        self.frame_count = 0


class HybridEventSystem:
    """
    Hybrid event detection system combining rule-based and ML detectors.
    
    The ML detector provides high-level event classification.
    The rule-based detector handles precise frame-level events.
    Both work together for comprehensive event detection.
    """
    
    def __init__(self, rule_detector, ml_detector, config):
        """
        Initialize hybrid system.
        
        Args:
            rule_detector: Existing RuleBasedEventDetector
            ml_detector: MLEventDetector instance
            config: Pipeline configuration
        """
        self.rule_detector = rule_detector
        self.ml_detector = ml_detector
        self.config = config
        
        # Event fusion settings
        self.ml_confidence_threshold = 0.7
        self.event_cooldown = 30  # Frames between same ML events
        self.last_ml_events = {}  # Track recent ML events
        
        logger.info("[Hybrid System] Initialized with rule + ML detectors")
    
    def detect_events(self, frame, ball_pos, player_tracks, frame_idx, fps):
        """
        Detect events using hybrid approach.
        
        Args:
            frame: Current frame
            ball_pos: Ball position
            player_tracks: Player tracking data
            frame_idx: Current frame index
            fps: Video FPS
        
        Returns:
            events: List of detected events
        """
        events = []
        
        # Get ML prediction
        ml_pred = self.ml_detector.predict()
        
        if ml_pred and ml_pred['confidence'] > self.ml_confidence_threshold:
            event_type = ml_pred['mapped_event']
            
            # Check cooldown to avoid duplicate events
            last_frame = self.last_ml_events.get(event_type, -999)
            if (frame_idx - last_frame) > self.event_cooldown:
                # Add ML event
                events.append({
                    'type': event_type,
                    'frame': frame_idx,
                    'timestamp': frame_idx / fps,
                    'confidence': ml_pred['confidence'],
                    'detector': 'ml',
                    'class': ml_pred['class']
                })
                
                self.last_ml_events[event_type] = frame_idx
                
                logger.info(
                    f"[ML Event] {ml_pred['class']} → {event_type} "
                    f"(conf: {ml_pred['confidence']:.2f}) at frame {frame_idx}"
                )
        
        return events


def integrate_ml_detector_into_pipeline(pipeline, weights_path):
    """
    Integrate ML detector into existing pipeline.
    
    This is the main integration function that modifies the pipeline
    to use both rule-based and ML detection.
    
    Args:
        pipeline: MatchAnalysisPipeline instance
        weights_path: Path to trained model weights
    
    Returns:
        success: True if integration successful
    """
    try:
        # Create ML detector
        ml_detector = MLEventDetector(
            weights_path=weights_path,
            config=pipeline.config,
            device='auto'
        )
        
        # Create hybrid system
        # Note: pipeline.event_detector is the existing RuleBasedEventDetector
        hybrid_system = HybridEventSystem(
            rule_detector=pipeline.event_detector,
            ml_detector=ml_detector,
            config=pipeline.config
        )
        
        # Store in pipeline
        pipeline.ml_detector = ml_detector
        pipeline.hybrid_event_system = hybrid_system
        
        logger.info("[Integration] ML event detector successfully integrated")
        return True
        
    except Exception as e:
        logger.error(f"[Integration] Failed: {e}")
        logger.warning("[Integration] Continuing with rule-based detection only")
        return False
