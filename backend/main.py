#!/usr/bin/env python3
"""
AI-Powered Match Analysis System for Low-Resource Football Games
================================================================
Main entry point for the analysis pipeline.

v4 updates:
- Added ML model integration support
- Support for all 18 event types
- Freeze frame generation

Usage:
    python main.py --input input/match.mp4 --output output/analysis.mp4
    python main.py --input input/match.mp4 --ml-model models/event_detector_weights.pth
    python main.py --demo  # Run with sample/demo video

Author: Muaad M F M (IT22323620)
Project: 25-26J-099
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, INPUT_DIR
from fastapi_app.services.analysis_service import AnalysisRequestOptions, AnalysisService


def setup_logging(verbose: bool = True):
    """Configure logging for the application."""
    level = logging.INFO if verbose else logging.WARNING

    # Console handler with safe encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # File handler with explicit UTF-8 encoding (prevents Windows cp1252 issues)
    file_handler = logging.FileHandler(
        os.path.join(OUTPUT_DIR, "pipeline.log"),
        mode="w",
        encoding="utf-8",
    )
    file_handler.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=level,
        handlers=[console_handler, file_handler],
    )


def create_demo_video(output_path: str, duration: int = 5, fps: int = 30) -> str:
    """
    Create a simple demo/test video with synthetic football-like scene.
    Used for testing the pipeline when no real video is available.
    """
    import cv2
    import numpy as np

    width, height = 960, 540
    total_frames = duration * fps

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Synthetic players (position, color, team)
    players = [
        # Team 0 (Red)
        {"pos": [200, 300], "vel": [2, 1], "color": (0, 0, 200), "team": 0},
        {"pos": [300, 200], "vel": [1, 2], "color": (0, 0, 200), "team": 0},
        {"pos": [350, 350], "vel": [3, -1], "color": (0, 0, 200), "team": 0},
        {"pos": [150, 250], "vel": [-1, 2], "color": (0, 0, 200), "team": 0},
        # Team 1 (Blue)
        {"pos": [600, 250], "vel": [-2, 1], "color": (200, 0, 0), "team": 1},
        {"pos": [700, 350], "vel": [-1, -2], "color": (200, 0, 0), "team": 1},
        {"pos": [650, 150], "vel": [-3, 1], "color": (200, 0, 0), "team": 1},
        {"pos": [750, 300], "vel": [1, -1], "color": (200, 0, 0), "team": 1},
        # Referee (Black)
        {"pos": [480, 270], "vel": [1, 0], "color": (30, 30, 30), "team": -1},
    ]

    ball_pos = [400.0, 270.0]
    ball_vel = [3.0, 1.5]

    for f in range(total_frames):
        # Green pitch
        frame = np.full((height, width, 3), (34, 139, 34), dtype=np.uint8)

        # Pitch lines
        cv2.rectangle(frame, (30, 30), (width - 30, height - 30), (255, 255, 255), 2)
        cv2.line(frame, (width // 2, 30), (width // 2, height - 30), (255, 255, 255), 2)
        cv2.circle(frame, (width // 2, height // 2), 60, (255, 255, 255), 2)

        # Goal areas
        cv2.rectangle(frame, (30, height // 4), (100, 3 * height // 4), (255, 255, 255), 2)
        cv2.rectangle(frame, (width - 100, height // 4), (width - 30, 3 * height // 4), (255, 255, 255), 2)

        # Move and draw players
        for p in players:
            p["pos"][0] += p["vel"][0] + np.random.randn() * 0.5
            p["pos"][1] += p["vel"][1] + np.random.randn() * 0.5

            # Bounce off edges
            if p["pos"][0] < 50 or p["pos"][0] > width - 50:
                p["vel"][0] *= -1
            if p["pos"][1] < 50 or p["pos"][1] > height - 50:
                p["vel"][1] *= -1

            px, py = int(p["pos"][0]), int(p["pos"][1])
            # Draw player as rectangle (simulating bounding box detection)
            cv2.rectangle(frame, (px - 12, py - 30), (px + 12, py + 30),
                          p["color"], -1)
            cv2.rectangle(frame, (px - 12, py - 30), (px + 12, py + 30),
                          (255, 255, 255), 1)

        # Move and draw ball
        ball_pos[0] += ball_vel[0]
        ball_pos[1] += ball_vel[1]
        if ball_pos[0] < 40 or ball_pos[0] > width - 40:
            ball_vel[0] *= -1
        if ball_pos[1] < 40 or ball_pos[1] > height - 40:
            ball_vel[1] *= -1

        bx, by = int(ball_pos[0]), int(ball_pos[1])
        cv2.circle(frame, (bx, by), 8, (255, 255, 255), -1)
        cv2.circle(frame, (bx, by), 8, (0, 0, 0), 2)

        writer.write(frame)

    writer.release()
    print(f"Demo video created: {output_path} ({total_frames} frames)")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI-Powered Match Analysis for Low-Resource Football Games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input match.mp4
  python main.py --input match.mp4 --output result.mp4 --no-reid
  python main.py --demo
  python main.py --input match.mp4 --ml-model models/event_detector_weights.pth
  python main.py --input match.mp4 --frame-skip 2 --confidence 0.4
        """,
    )

    # Input/Output
    parser.add_argument("--input", "-i", type=str, default="",
                        help="Path to input video file")
    parser.add_argument("--output", "-o", type=str, default="",
                        help="Path for output video (default: output/analysis_<input>.mp4)")
    parser.add_argument("--json-output", type=str, default="",
                        help="Path for JSON analysis output")
    parser.add_argument("--demo", action="store_true",
                        help="Run with a synthetic demo video")

    # Detection
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model name/path (default: yolov8n.pt)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Detection confidence threshold (default: 0.3)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for inference")

    # Processing
    parser.add_argument("--frame-skip", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--max-width", type=int, default=1280,
                        help="Maximum input width (default: 1280)")
    parser.add_argument("--max-height", type=int, default=720,
                        help="Maximum input height (default: 720)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Maximum frames to process (0=all)")

    # Feature toggles
    parser.add_argument("--no-stabilization", action="store_true",
                        help="Disable video stabilization")
    parser.add_argument("--no-reid", action="store_true",
                        help="Disable player re-identification")
    parser.add_argument("--no-events", action="store_true",
                        help="Disable ML-based event detection")
    parser.add_argument("--no-minimap", action="store_true",
                        help="Disable minimap overlay")
    parser.add_argument("--no-freeze-frames", action="store_true",
                        help="Disable freeze frame generation")

    # v4: ML Model options
    parser.add_argument("--ml-model", type=str, default="",
                        help="Path to trained ML event detector weights (e.g., models/event_detector_weights.pth)")
    parser.add_argument("--ml-confidence", type=float, default=0.7,
                        help="ML event detection confidence threshold (default: 0.7)")
    parser.add_argument("--ml-device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device for ML model inference")

    # Other
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    ml_model_enabled = False
    ml_model_path = ""
    if args.ml_model:
        if not os.path.exists(args.ml_model):
            print(f"Warning: ML model file not found: {args.ml_model}")
            print("Continuing with rule-based detection only")
        else:
            ml_model_enabled = True
            ml_model_path = args.ml_model

    options = AnalysisRequestOptions(
        model=args.model,
        confidence=args.confidence,
        device=args.device,
        frame_skip=args.frame_skip,
        max_width=args.max_width,
        max_height=args.max_height,
        enable_stabilization=not args.no_stabilization,
        enable_reid=not args.no_reid,
        enable_events=not args.no_events,
        enable_minimap=not args.no_minimap,
        enable_freeze_frames=not args.no_freeze_frames,
        quiet=args.quiet,
        enable_ml_detector=ml_model_enabled,
        ml_model_path=ml_model_path or None,
        ml_confidence=args.ml_confidence,
        ml_device=args.ml_device,
    )

    setup_logging(verbose=not options.quiet)

    # Handle demo mode
    if args.demo:
        demo_path = os.path.join(INPUT_DIR, "demo_match.mp4")
        args.input = create_demo_video(demo_path, duration=10, fps=30)

    # Validate input
    if not args.input:
        print("Error: --input is required (or use --demo)")
        print("Usage: python main.py --input <video_path>")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        sys.exit(1)

    # Set output paths
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    if not args.output:
        args.output = os.path.join(OUTPUT_DIR, f"analysis_{input_basename}.mp4")
    if not args.json_output:
        args.json_output = os.path.join(OUTPUT_DIR, f"analysis_{input_basename}.json")

    # Print configuration summary
    print("="*60)
    print("Match Analysis Configuration")
    print("="*60)
    print(f"Input Video: {args.input}")
    print(f"Output Video: {args.output}")
    print(f"YOLO Model: {options.model}")
    print(f"Frame Skip: {options.frame_skip}")
    print(f"Re-ID: {'Enabled' if options.enable_reid else 'Disabled'}")
    print(f"ML Event Detector: {'Enabled' if options.enable_ml_detector else 'Disabled'}")
    if options.enable_ml_detector:
        print(f"  Model Weights: {options.ml_model_path}")
        print(f"  Confidence: {options.ml_confidence}")
        print(f"  Device: {options.ml_device}")
    print(f"Freeze Frames: {'Enabled' if options.enable_freeze_frames else 'Disabled'}")
    print("="*60)

    # Run pipeline via shared analysis service
    analysis_service = AnalysisService()
    results = analysis_service.run_analysis_with_paths(
        input_video=Path(args.input),
        output_video=Path(args.output),
        output_json=Path(args.json_output),
        options=options,
    )

    print(f"\n{'='*60}")
    print("Analysis Complete!")
    print(f"  Output video: {results['output_video']}")
    print(f"  JSON report:  {results['json_report']}")
    print(f"  Frames:       {results['frames_processed']}")
    print(f"  Avg FPS:      {results['avg_fps']:.1f}")
    print(f"  Events:       {results['events_detected']}")
    if results.get('ml_detector_used'):
        print("  ML Detector:  Active")
    print(f"  Possession:   Team 0: {results['possession'][0]:.1f}% | "
          f"Team 1: {results['possession'][1]:.1f}%")

    # v4: Show event breakdown
    print("\n  Event Types Detected:")
    event_summary = results.get('event_summary', {})
    for event_type, count in sorted(event_summary.items(), key=lambda x: x[1], reverse=True):
        print(f"    {event_type}: {count}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
