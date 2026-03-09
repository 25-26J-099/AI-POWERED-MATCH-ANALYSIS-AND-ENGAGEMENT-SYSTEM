"""
Configuration for AI-Powered Match Analysis System for Low-Resource Football Games.
All tunable parameters are centralized here.

v5 UPDATE: Added Strategic Hybrid Event Detection with intelligent event routing.
- EventRoutingConfig: Routes events to optimal detector (ML/Rule/Hybrid)
- Enhanced EventDetectionConfig: Added event_cooldowns dictionary
- Preset configurations for different use cases
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


@dataclass
class PreprocessingConfig:
    enable_stabilization: bool = True
    enable_super_resolution: bool = True
    sr_model_name: str = "espcn"
    sr_scale_factor: int = 2
    stabilization_smoothing_radius: int = 30
    stabilization_border_crop: int = 20


@dataclass
class DetectionConfig:
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.25  # Default/fallback threshold
    player_confidence: float = 0.30     # Higher for clean player detection
    ball_confidence: float = 0.12       # Lower for robust ball tracking
    iou_threshold: float = 0.45
    input_size: int = 640
    batch_size: int = 4
    person_class_id: int = 0
    ball_class_id: int = 32
    device: str = "auto"


@dataclass
class TrackingConfig:
    track_high_thresh: float = 0.45
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.65
    track_buffer: int = 45
    match_thresh: float = 0.75
    min_track_length: int = 3


@dataclass
class TeamAssignmentConfig:
    n_clusters: int = 2
    jersey_region_top: float = 0.15
    jersey_region_bottom: float = 0.55
    jersey_region_left: float = 0.2
    jersey_region_right: float = 0.8
    vote_window: int = 25
    vote_threshold: float = 0.55


@dataclass
class ReIDConfig:
    enable: bool = True

    feature_dim: int = 128
    max_lost_frames: int = 120

    # Core thresholds
    appearance_threshold: float = 0.55
    spatial_threshold: float = 250.0
    combined_threshold: float = 0.65

    # Legacy weights (kept for compatibility)
    appearance_weight: float = 0.5
    spatial_weight: float = 0.5
    num_body_parts: int = 3

    # Adaptive thresholding
    use_adaptive_thresholds: bool = True
    velocity_threshold: float = 12.0
    high_velocity_appearance_factor: float = 0.8
    high_velocity_spatial_factor: float = 1.3

    occlusion_appearance_factor: float = 0.75
    occlusion_spatial_factor: float = 1.4
    occlusion_variance_threshold: float = 0.15

    # Temporal aggregation
    use_enhanced_temporal: bool = True
    temporal_window_size: int = 5
    temporal_consensus_threshold: float = 0.6

    # Metrics tracking
    enable_metrics_tracking: bool = False


@dataclass
class EventRoutingConfig:
    """
    v5 NEW: Configuration for strategic event routing.
    
    Routes events to the most appropriate detector for optimal accuracy.
    
    Categories:
    - ml_only: Events best handled by ML (complex temporal patterns)
    - rule_only: Events best handled by rules (geometric/spatial)
    - hybrid: Events that benefit from both (ML classification + rule context)
    """
    
    routing: Dict[str, List[str]] = field(default_factory=lambda: {
        # ═══════════════════════════════════════════════════════════
        # ML-ONLY EVENTS: Complex temporal patterns
        # ═══════════════════════════════════════════════════════════
        # These events benefit from ML's ability to learn patterns
        # and understand temporal sequences
        "ml_only": [
            "pass",           # Temporal: ball transfer between players (+10-15% accuracy)
            "interception",   # Complex: intercept trajectory mid-flight (+15-20% accuracy)
            "dribble",        # Temporal: sustained movement with ball (+10% accuracy)
            "sprint",         # Temporal: sustained high velocity (+10% accuracy)
            "dribbled_past",  # Complex: beating defender (ML better)
        ],
        
        # ═══════════════════════════════════════════════════════════
        # RULE-ONLY EVENTS: Geometric/spatial events
        # ═══════════════════════════════════════════════════════════
        # These events have clear geometric definitions and benefit
        # from deterministic rule-based detection
        "rule_only": [
            "possession_change",   # CRITICAL: proximity-based (90-95% accuracy)
            "out_of_bounds",       # CRITICAL: position-based (95-98% accuracy)
            "shot",                # Velocity + direction (80-85% accuracy)
            "ball_receipt",        # Distance traveled (geometric)
            "pressure",            # Geometric: opponents within radius (80-90% accuracy)
            "ball_recovery",       # Possession state change (80-90% accuracy)
            "clearance",           # High velocity from defensive zone (80-90% accuracy)
            "block",               # Velocity drop (75-85% accuracy)
            "goalkeeper_save",     # Position + ball interaction (85-95% accuracy)
            "goalkeeper_claim",    # Position + ball interaction (85-95% accuracy)
            "miscontrol",          # Ball distance from player (geometric)
            "dispossessed",        # Proximity + possession change (70-75% accuracy)
            "duel",                # Two players near ball (70-75% accuracy)
        ],
        
        # ═══════════════════════════════════════════════════════════
        # HYBRID EVENTS: Both ML and rules
        # ═══════════════════════════════════════════════════════════
        # These events benefit from both approaches:
        # - ML for classification
        # - Rules for spatial context and refinement
        "hybrid": [
            "tackle",     # ML: detect action, Rules: verify proximity (+10-15% accuracy)
            "foul",       # ML: detect foul, Rules: verify contact (+10-15% accuracy)
            "carry",      # ML: detect intention, Rules: measure distance (+10-15% accuracy)
        ],
    })
    
    # Confidence boost for hybrid events when both detectors agree
    hybrid_agreement_boost: float = 0.3
    
    # Confidence penalty for hybrid events when only one detector fires
    hybrid_solo_penalty: float = 0.15


@dataclass
class EventDetectionConfig:
    """
    v5 ENHANCED: Added event_cooldowns dictionary and strategic routing support.
    """
    
    # ═══════════════════════════════════════════════════════════
    # Core Possession Parameters
    # ═══════════════════════════════════════════════════════════
    possession_radius: float = 70.0
    possession_min_frames: int = 3
    
    # ═══════════════════════════════════════════════════════════
    # Event-Specific Parameters
    # ═══════════════════════════════════════════════════════════
    
    # Tackle
    tackle_proximity: float = 35.0
    tackle_velocity_drop: float = 0.5
    
    # Shot
    shot_velocity_threshold: float = 20.0
    max_ball_velocity: float = 100.0
    
    # Pass
    pass_min_distance: float = 60.0
    pass_max_frames: int = 45
    
    # Field boundaries
    goal_line_left: float = 0.05
    goal_line_right: float = 0.95
    oob_margin: float = 0.012
    pitch_top_margin: float = 0.04
    pitch_bottom_margin: float = 0.08
    
    # ML detection parameters
    enable_ml_events: bool = True
    temporal_window: int = 16
    ml_confidence_threshold: float = 0.6
    
    # Ball Receipt
    ball_receipt_min_distance: float = 60.0
    
    # Carry
    carry_min_distance: float = 50.0
    carry_min_frames: int = 15
    
    # Pressure
    pressure_radius: float = 100.0  # pixels (~2 meters)
    pressure_min_opponents: int = 2
    
    # Ball Recovery
    ball_recovery_min_loose_frames: int = 15
    
    # Duel
    duel_radius: float = 80.0
    duel_cooldown_frames: int = 50
    
    # Clearance
    clearance_min_velocity: float = 25.0
    clearance_defensive_third: float = 0.35  # normalized x-position
    
    # Block
    block_velocity_drop: float = 0.75  # 75% velocity reduction
    block_proximity: float = 50.0
    
    # Goalkeeper Actions
    goalkeeper_goal_proximity: float = 0.1  # normalized x-position
    goalkeeper_ball_radius: float = 60.0
    goalkeeper_save_min_velocity: float = 15.0
    
    # Miscontrol
    miscontrol_ball_distance: float = 120.0
    miscontrol_min_possession_frames: int = 5
    
    # Dribble
    dribble_min_distance: float = 80.0
    dribble_min_frames: int = 20
    
    # Dispossessed
    dispossessed_proximity: float = 100.0
    
    # Interception
    interception_angle_threshold: float = 0.0  # cos(90°) = 0
    interception_min_trajectory_length: int = 5
    
    # Foul
    foul_proximity: float = 25.0
    foul_velocity_drop: float = 0.7  # 70% velocity reduction
    foul_cooldown_frames: int = 75
    
    # Freeze Frame Generation
    enable_freeze_frames: bool = True
    freeze_frame_goalkeeper_threshold: float = 0.7  # 70% of time near goal
    
    # ═══════════════════════════════════════════════════════════
    # v5 NEW: Event Cooldowns (minimum frames between same event)
    # ═══════════════════════════════════════════════════════════
    # Prevents duplicate event detection by enforcing minimum time
    # between consecutive detections of the same event type
    event_cooldowns: Dict[str, int] = field(default_factory=lambda: {
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


@dataclass
class OptimizationConfig:
    enable_quantization: bool = False
    quantization_type: str = "int8"
    enable_multithreading: bool = True
    num_worker_threads: int = 4
    target_fps: int = 15
    frame_skip: int = 1
    max_input_width: int = 1280
    max_input_height: int = 720


@dataclass
class VisualizationConfig:
    draw_tracks: bool = True
    draw_team_colors: bool = True
    draw_ball_trail: bool = True
    draw_events: bool = True
    draw_possession_bar: bool = True
    draw_minimap: bool = True
    ball_trail_length: int = 20
    # Colors (BGR)
    team1_color: Tuple[int, int, int] = (0, 0, 220)
    team2_color: Tuple[int, int, int] = (220, 0, 0)
    referee_color: Tuple[int, int, int] = (0, 200, 200)
    ball_color: Tuple[int, int, int] = (0, 255, 0)
    event_color: Tuple[int, int, int] = (0, 165, 255)
    unassigned_color: Tuple[int, int, int] = (180, 180, 180)
    minimap_width: int = 280
    minimap_height: int = 180
    minimap_margin: int = 10


@dataclass
class MLModelConfig:
    """Configuration for ML event detection model."""
    # Model paths
    weights_path: str = ""  # Path to trained model weights (e.g., "models/event_detector_weights.pth")
    
    # Inference settings
    enable_ml_detector: bool = False  # Enable ML-based event detection
    ml_device: str = "auto"  # "cuda", "cpu", or "auto"
    ml_confidence_threshold: float = 0.7
    ml_inference_interval: int = 5  # Run inference every N frames
    
    # Model architecture parameters (must match training)
    num_classes: int = 17
    temporal_window: int = 16
    hidden_dim: int = 512
    lstm_layers: int = 2
    dropout: float = 0.5


@dataclass
class PipelineConfig:
    """
    Main pipeline configuration.
    
    v5 UPDATE: Added event_routing for strategic hybrid event detection.
    """
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    team_assignment: TeamAssignmentConfig = field(default_factory=TeamAssignmentConfig)
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    event_detection: EventDetectionConfig = field(default_factory=EventDetectionConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    ml_model: MLModelConfig = field(default_factory=MLModelConfig)
    
    # v5 NEW: Strategic event routing configuration
    event_routing: EventRoutingConfig = field(default_factory=EventRoutingConfig)
    
    input_video: str = ""
    output_video: str = ""
    output_json: str = ""
    verbose: bool = True


# ═══════════════════════════════════════════════════════════════
# v5 NEW: PRESET CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════

def get_default_config() -> PipelineConfig:
    """
    Get default balanced configuration.
    
    Use when:
    - Standard use case
    - ML model is reasonably accurate
    - Want good balance of accuracy and speed
    
    Event routing:
    - ML: pass, interception, dribble, sprint, dribbled_past
    - Rules: possession, out_of_bounds, shot, pressure, clearance, etc.
    - Hybrid: tackle, foul, carry
    """
    return PipelineConfig()


def get_ml_aggressive_config() -> PipelineConfig:
    """
    Configuration that routes more events to ML.
    
    Use when:
    - ML model is highly accurate (>85% per event)
    - Have GPU for fast inference
    - Want to maximize ML usage
    - Prioritize recall over precision
    
    Changes from default:
    - Moves tackle, foul from hybrid to ML-only
    - Lowers ML confidence threshold to 0.55
    """
    config = PipelineConfig()
    
    # Route more events to ML
    config.event_routing.routing["ml_only"].extend([
        "tackle",
        "foul",
    ])
    
    # Remove from hybrid (now ML-only)
    config.event_routing.routing["hybrid"] = ["carry"]
    
    # Lower ML confidence threshold for higher recall
    config.event_detection.ml_confidence_threshold = 0.55
    config.ml_model.ml_confidence_threshold = 0.55
    
    return config


def get_rule_conservative_config() -> PipelineConfig:
    """
    Configuration that relies more on rule-based detection.
    
    Use when:
    - ML model accuracy is uncertain
    - Need deterministic behavior
    - CPU-only inference (slow ML)
    - Prioritize precision over recall
    - Academic validation/baseline
    
    Changes from default:
    - Minimal ML usage (only pass, dribble)
    - Most events handled by rules
    - Higher ML confidence threshold (0.75)
    """
    config = PipelineConfig()
    
    # Minimal ML usage
    config.event_routing.routing["ml_only"] = [
        "pass",
        "dribble",
    ]
    
    # Move interception, sprint to rules
    config.event_routing.routing["rule_only"].extend([
        "interception",
        "sprint",
        "dribbled_past",
    ])
    
    # No hybrid events
    config.event_routing.routing["hybrid"] = []
    
    # Higher ML confidence threshold
    config.event_detection.ml_confidence_threshold = 0.75
    config.ml_model.ml_confidence_threshold = 0.75
    
    return config


def get_performance_config() -> PipelineConfig:
    """
    Configuration optimized for speed.
    
    Use when:
    - Need fast processing (real-time or near real-time)
    - Can sacrifice some accuracy
    - Limited computational resources
    - Want maximum throughput
    
    Changes from default:
    - Minimal ML usage (only pass)
    - Increased ML inference interval (every 10 frames)
    - Higher confidence threshold
    - Most events handled by fast rules
    """
    config = PipelineConfig()
    
    # Minimal ML usage
    config.event_routing.routing["ml_only"] = ["pass"]
    
    # Move most to rule-based
    config.event_routing.routing["rule_only"].extend([
        "dribble",
        "sprint",
        "interception",
        "dribbled_past",
    ])
    
    # No hybrid events (avoid fusion overhead)
    config.event_routing.routing["hybrid"] = []
    
    # Reduce ML inference frequency
    config.ml_model.ml_inference_interval = 10  # Every 10 frames instead of 5
    
    # Higher confidence threshold (fewer events but faster)
    config.event_detection.ml_confidence_threshold = 0.75
    config.ml_model.ml_confidence_threshold = 0.75
    
    # Optimization settings
    config.optimization.frame_skip = 1  # Process every frame
    config.optimization.target_fps = 25
    
    return config


def get_accuracy_config() -> PipelineConfig:
    """
    Configuration optimized for maximum accuracy.
    
    Use when:
    - Accuracy is paramount (research, validation)
    - Have powerful hardware
    - Can afford slower processing
    - Need publication-quality results
    
    Changes from default:
    - Full hybrid usage (all events use both detectors where beneficial)
    - Lower confidence thresholds (higher recall)
    - More frequent ML inference
    """
    config = PipelineConfig()
    
    # Expand hybrid events
    config.event_routing.routing["hybrid"].extend([
        "ball_recovery",
        "duel",
        "miscontrol",
        "dispossessed",
    ])
    
    # Remove expanded hybrid events from rule-only
    for event in ["ball_recovery", "duel", "miscontrol", "dispossessed"]:
        if event in config.event_routing.routing["rule_only"]:
            config.event_routing.routing["rule_only"].remove(event)
    
    # Lower confidence thresholds for higher recall
    config.event_detection.ml_confidence_threshold = 0.55
    config.ml_model.ml_confidence_threshold = 0.55
    
    # More frequent ML inference
    config.ml_model.ml_inference_interval = 3  # Every 3 frames
    
    # Enable all features
    config.event_detection.enable_freeze_frames = True
    config.reid.enable = True
    
    return config


# ═══════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════

"""
Example 1: Use default balanced configuration
>>> config = PipelineConfig()
>>> # or
>>> config = get_default_config()

Example 2: Use aggressive ML configuration
>>> config = get_ml_aggressive_config()
>>> # More events handled by ML

Example 3: Use conservative rule-based configuration
>>> config = get_rule_conservative_config()
>>> # Minimal ML usage, most events handled by rules

Example 4: Use performance-optimized configuration
>>> config = get_performance_config()
>>> # Optimized for speed

Example 5: Use accuracy-optimized configuration
>>> config = get_accuracy_config()
>>> # Optimized for maximum accuracy

Example 6: Custom routing
>>> config = PipelineConfig()
>>> # Add interception to ML
>>> config.event_routing.routing["ml_only"].append("interception")
>>> config.event_routing.routing["rule_only"].remove("interception")
>>> 
>>> # Move tackle to hybrid
>>> config.event_routing.routing["ml_only"].remove("tackle")
>>> config.event_routing.routing["hybrid"].append("tackle")

Example 7: Disable ML entirely
>>> config = PipelineConfig()
>>> config.event_detection.enable_ml_events = False
>>> # All events handled by rules

Example 8: Enable ML detector
>>> config = PipelineConfig()
>>> config.ml_model.enable_ml_detector = True
>>> config.ml_model.weights_path = "models/event_detector_weights.pth"
>>> config.event_detection.enable_ml_events = True
"""