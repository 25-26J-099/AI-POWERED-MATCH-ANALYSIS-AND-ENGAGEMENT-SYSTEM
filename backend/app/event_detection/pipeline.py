"""
AI-Powered Match Analysis Pipeline for Low-Resource Football Games.

v4 updates:
- Integrated ML event detector (SoccerNet-trained model)
- Added freeze frame generation for all events
- Support for all 18 event types
- Enhanced event enrichment
"""
import cv2
import numpy as np
import logging
import time
import os
from typing import Optional, Dict

from app.config.pipeline_config import PipelineConfig
from app.config.settings import settings
from app.event_detection.video_preprocessor import VideoPreprocessor
from app.event_detection.tracker import PlayerBallTracker
from app.event_detection.team_assigner import TeamAssigner
from app.event_detection.player_reid import PlayerReIDModule
from app.event_detection.strategic_hybrid_detector import StrategicHybridEventDetector
from app.event_detection.statsbomb_export import StatsBombExporter
from app.services.model_loader import download_hf_asset
from app.utils.video_utils import VideoReader, VideoWriter
from app.utils.drawing_utils import Annotator
from app.utils.data_export import MatchDataExporter

from app.event_detection.robust_reid import RobustReIDSystem

# v4: Import ML event detector
try:
    from app.models.ml_event_detector import MLEventDetector, integrate_ml_detector_into_pipeline
    ML_AVAILABLE = True
except ImportError:
    MLEventDetector = None
    integrate_ml_detector_into_pipeline = None
    ML_AVAILABLE = False
    logging.warning("ML event detector not available - continuing with rule-based detection only")

logger = logging.getLogger(__name__)


def _pf(v):
    """Convert any numeric to plain Python float."""
    return float(v) if v is not None else 0.0


class MatchAnalysisPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.preprocessor = VideoPreprocessor(config)
        self.tracker = PlayerBallTracker(config)
        self.team_assigner = TeamAssigner(config)
        self.reid_module = PlayerReIDModule(config)
        self.robust_reid = RobustReIDSystem(config)
        self.event_detector = StrategicHybridEventDetector(config)
        self.statsbomb_exporter = StatsBombExporter()
        self.annotator = Annotator(config)

        output_dir = os.path.dirname(config.output_json) if config.output_json else \
            os.path.dirname(config.output_video) if config.output_video else "output"
        if not output_dir:
            output_dir = "output"
        self.exporter = MatchDataExporter(output_dir)

        self.frame_count = 0
        self.fps = 30.0
        self._team_fitted = False
        self._sample_count = 0
        
        # v4: ML detector integration
        self.ml_detector = None
        self.hybrid_event_system = None
        self._ml_enabled = False
        
        # Initialize ML detector if configured
        if ML_AVAILABLE and config.ml_model.enable_ml_detector:
            self._initialize_ml_detector()

    def _initialize_ml_detector(self):
        """Initialize ML event detector if available and configured."""
        if not ML_AVAILABLE:
            logger.warning("ML event detector module not available")
            return
        
        try:
            weights_path = self.config.ml_model.weights_path
            if not os.path.exists(weights_path):
                requested_filename = os.path.basename(weights_path) if weights_path else settings.HF_EVENT_DETECTOR_WEIGHTS_FILE
                downloaded_path = download_hf_asset(
                    settings.HF_FOOTBALL_MODELS_REPO,
                    requested_filename or settings.HF_EVENT_DETECTOR_WEIGHTS_FILE,
                )
                if downloaded_path:
                    weights_path = downloaded_path
                    logger.info("[Pipeline] ML model weights downloaded from HuggingFace: %s", weights_path)
                else:
                    logger.warning(f"ML model weights not found locally or on HuggingFace: {weights_path}")
                    return
            
            # Integrate ML detector
            success = integrate_ml_detector_into_pipeline(self, weights_path)
            
            if success:
                self._ml_enabled = True
                logger.info("[Pipeline] ML event detector successfully integrated")
            else:
                logger.warning("[Pipeline] ML detector integration failed - using rule-based only")
                
        except Exception as e:
            logger.error(f"[Pipeline] Error initializing ML detector: {e}")
            logger.warning("[Pipeline] Continuing with rule-based detection only")

    def initialize(self):
        logger.info("=" * 60)
        logger.info("Initializing Match Analysis Pipeline")
        logger.info("=" * 60)
        self.tracker.initialize()
        logger.info("[OK] Tracker initialized")
        
        if self._ml_enabled:
            logger.info("[OK] ML event detector active")
        else:
            logger.info("[OK] Rule-based event detection only")

    def _collect_and_fit_teams(self, frame, tracks):
        """Collect jersey samples and fit team model when ready."""
        if self._team_fitted:
            return
        fh, fw = frame.shape[:2]
        for tid, t in tracks.items():
            if not t.is_ball and not t.is_referee:
                cx, cy = t.center
                if (cx > fw * 0.05 and cx < fw * 0.95
                        and cy > fh * 0.05 and cy < fh * 0.85):
                    self.team_assigner.collect_sample(frame, t.bbox)
                    self._sample_count += 1
        if self._sample_count >= 30 and not self._team_fitted:
            if self.team_assigner.fit():
                self._team_fitted = True
                logger.info("[OK] Team assignment model fitted")
                auto_colors = self.team_assigner.get_team_display_colors()
                if auto_colors:
                    self.annotator.set_team_colors(auto_colors)
                    for tid, color in auto_colors.items():
                        logger.info(f"Team {tid} display color (BGR): {color}")

    def _assign_teams_continuous(self, frame, tracks):
        """Assign/reassign teams for ALL active players every frame."""
        if not self._team_fitted:
            return
        fh, fw = frame.shape[:2]
        for tid, track in tracks.items():
            if track.is_ball:
                continue

            # Pitch ROI filter
            cx, cy = track.center
            if cy < fh * 0.05 or cy > fh * 0.92:
                track.team_id = -1
                continue

            # Vote-based referee check
            if track.frames_tracked % 10 == 1:
                is_ref = self.team_assigner.classify_referee(frame, track.bbox, tid)
                track.is_referee = is_ref
                if is_ref:
                    track.team_id = -1
                    continue

            if track.is_referee:
                continue

            track.team_id = self.team_assigner.assign_team(frame, track.bbox, tid)

    def process_video(self, input_path: str, output_path: str, json_path: str = ""):
        reader = VideoReader(
            input_path,
            frame_skip=self.config.optimization.frame_skip,
            max_width=self.config.optimization.max_input_width,
            max_height=self.config.optimization.max_input_height,
        )
        self.fps = reader.fps
        writer = VideoWriter(output_path, reader.width, reader.height, fps=reader.fps)

        self.exporter.set_metadata({
            "input_video": input_path,
            "resolution": f"{reader.width}x{reader.height}",
            "fps": reader.fps,
            "total_frames": reader.total_frames,
            "duration_seconds": reader.duration,
            "ml_detector_enabled": self._ml_enabled,
        })
        self.statsbomb_exporter.set_frame_dimensions(reader.width, reader.height)

        start_time = time.time()
        total_proc = 0

        logger.info(f"Processing: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Frames: {reader.total_frames} | FPS: {reader.fps:.1f}")
        logger.info(f"ML Detector: {'Enabled' if self._ml_enabled else 'Disabled'}")
        logger.info("-" * 60)

        try:
            for frame_idx, frame in reader.read_frames():
                t0 = time.time()

                # Module 1: Preprocessing
                processed = self.preprocessor.process_single_frame(frame)

                # Module 2: Detection + Tracking
                player_tracks, ball_track = self.tracker.process_frame(processed, frame_idx)

                # Team fitting (first ~30 frames)
                self._collect_and_fit_teams(frame, player_tracks)

                # Continuous team assignment (EVERY frame, not cached)
                self._assign_teams_continuous(frame, player_tracks)

                # Module 2.5: ROBUST RE-ID - Map ByteTrack IDs to Stable IDs
                player_tracks = self.robust_reid.process_frame(frame, player_tracks, frame_idx)

                # Module 3: Re-ID (legacy, kept for lost track gallery)
                if self.config.reid.enable and self._team_fitted:
                    lost = self.tracker.get_lost_tracks()
                    self.reid_module.update_gallery(
                        frame, player_tracks, lost, frame_idx
                    )
                
                # v4: Module 3.5: ML Event Detection (if enabled)
                if self._ml_enabled and self.ml_detector:
                    # Update ML detector with current frame
                    self.ml_detector.update_buffer(frame)
                    
                    # Get ML events (handled by hybrid system)
                    if self.hybrid_event_system:
                        ball_pos = ball_track.center if ball_track and ball_track.frames_lost < 5 else None
                        ml_events = self.hybrid_event_system.detect_events(
                            frame, ball_pos, player_tracks, frame_idx, self.fps
                        )
                        # ML events are added to event detector's event list
                        for ml_event in ml_events:
                            # Convert to GameEvent format
                            from app.event_detection.event_detector import GameEvent
                            event = GameEvent(
                                event_type=ml_event['type'],
                                frame_idx=ml_event['frame'],
                                timestamp=ml_event['timestamp'],
                                confidence=ml_event['confidence'],
                                source='ml'
                            )
                            self.event_detector.events.append(event)

                # Module 4: Event Detection (rule-based + ML hybrid)
                events = self.event_detector.process_frame(
                    processed, player_tracks, ball_track, frame_idx, self.fps
                )

                # ── Data export ────────────────────────────────────
                for tid, track in player_tracks.items():
                    if not track.is_ball:
                        self.exporter.add_player_position(
                            frame_idx, tid, track.bbox,
                            (_pf(track.center[0]), _pf(track.center[1])),
                            track.team_id,
                            _pf(track.velocity),
                        )

                if ball_track and ball_track.frames_lost < 5:
                    self.exporter.add_ball_position(
                        frame_idx,
                        (_pf(ball_track.center[0]), _pf(ball_track.center[1])),
                    )
                    poss = self.event_detector.rule.possession_player
                    poss_team = self.event_detector.rule.possession_team
                    if poss is not None and poss_team >= 0:
                        self.exporter.add_possession(frame_idx, poss_team, poss)

                # v4: Export events with freeze frames
                for ev in events:
                    self.exporter.add_event(ev.to_dict())

                # ── Visualization ──────────────────────────────────
                annotated = frame.copy()

                for tid, track in player_tracks.items():
                    if track.is_ball:
                        continue
                    annotated = self.annotator.draw_player(
                        annotated, track.bbox, tid,
                        team_id=track.team_id,
                        is_referee=track.is_referee,
                    )

                if ball_track and ball_track.frames_lost < 10:
                    annotated = self.annotator.draw_ball(annotated, ball_track.center)

                for ev in events:
                    annotated = self.annotator.draw_event(annotated, ev.to_dict(), frame_idx)

                t0_pct, t1_pct = self.event_detector.get_possession_stats()
                annotated = self.annotator.draw_possession_bar(annotated, t0_pct, t1_pct)

                if self.config.visualization.draw_minimap:
                    positions = {
                        tid: (_pf(t.center[0]), _pf(t.center[1]))
                        for tid, t in player_tracks.items() if not t.is_ball
                    }
                    teams = {tid: t.team_id for tid, t in player_tracks.items() if not t.is_ball}
                    ball_pos = (
                        (_pf(ball_track.center[0]), _pf(ball_track.center[1]))
                        if ball_track and ball_track.frames_lost < 5 else None
                    )
                    annotated = self.annotator.draw_minimap(annotated, positions, teams, ball_pos)

                stats_hud = {
                    "events_total": len(self.event_detector.events),
                    "players_detected": len(player_tracks),
                }
                if self._ml_enabled:
                    stats_hud["ml_detector"] = "ON"
                annotated = self.annotator.draw_stats_hud(annotated, frame_idx, self.fps, stats_hud)

                writer.write(annotated)
                self.frame_count += 1
                total_proc += time.time() - t0

                if frame_idx % 100 == 0 and frame_idx > 0:
                    avg_fps = self.frame_count / total_proc
                    pct = frame_idx / reader.total_frames * 100 if reader.total_frames else 0
                    logger.info(
                        f"Frame {frame_idx}/{reader.total_frames} ({pct:.1f}%) | "
                        f"{avg_fps:.1f} FPS | Players: {len(player_tracks)} | "
                        f"Events: {len(self.event_detector.events)}"
                    )

        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
        finally:
            writer.release()
            reader.release()

        elapsed = time.time() - start_time
        avg_fps = self.frame_count / total_proc if total_proc > 0 else 0

        logger.info("=" * 60)
        logger.info("Processing Complete")
        logger.info(f"  Frames processed: {self.frame_count}")
        logger.info(f"  Total time: {elapsed:.1f}s")
        logger.info(f"  Average FPS: {avg_fps:.1f}")
        logger.info(f"  Events detected: {len(self.event_detector.events)}")
        logger.info(f"  Players tracked: {len(self.exporter.player_tracks)}")

        # Export
        if json_path:
            jf = self.exporter.export_json(os.path.basename(json_path))
        else:
            jf = self.exporter.export_json()
        self.exporter.export_events_json()
        self.exporter.export_player_trajectories_csv()
        statsbomb_path = os.path.join(os.path.dirname(jf), "statsbomb_events.json")
        self.statsbomb_exporter.export_to_file(self.event_detector.events, statsbomb_path)

        summary = self.event_detector.get_event_summary()
        logger.info("\nEvent Summary:")
        for et, cnt in sorted(summary.items()):
            logger.info(f"  {et}: {cnt}")

        poss = self.event_detector.get_possession_stats()
        logger.info(f"\nPossession: Team 0: {poss[0]:.1f}% | Team 1: {poss[1]:.1f}%")
        logger.info(f"\nOutput video: {output_path}")
        logger.info(f"Analysis data: {jf}")
        logger.info(f"StatsBomb data: {statsbomb_path}")
        logger.info("=" * 60)

        return {
            "output_video": output_path,
            "json_report": jf,
            "frames_processed": self.frame_count,
            "processing_time": elapsed,
            "avg_fps": avg_fps,
            "events_detected": len(self.event_detector.events),
            "event_summary": summary,
            "possession": poss,
            "ml_detector_used": self._ml_enabled,
            "statsbomb_report": statsbomb_path,
        }
