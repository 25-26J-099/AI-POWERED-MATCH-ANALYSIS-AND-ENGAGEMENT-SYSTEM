"""
Data export utilities for structured match data output (JSON, CSV).

v5: Event export now uses JSON (events.json) instead of CSV.
"""
import json
import csv
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class _NumpySafeEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class MatchDataExporter:
    """Exports match analysis data to structured formats."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Accumulators
        self.events: List[dict] = []
        self.player_tracks: Dict[int, List[dict]] = {}
        self.ball_positions: List[dict] = []
        self.possession_log: List[dict] = []
        self.match_metadata: dict = {}

    def set_metadata(self, metadata: dict):
        """Set match metadata (video info, config, etc.)."""
        self.match_metadata = {
            "analysis_timestamp": datetime.now().isoformat(),
            **metadata,
        }

    def add_event(self, event: dict):
        """Add a detected event (v4: with freeze frame support)."""
        self.events.append(event)

    def add_player_position(
        self,
        frame_idx: int,
        track_id: int,
        bbox: tuple,
        position: tuple,
        team_id: int = -1,
        velocity: float = 0.0,
    ):
        """Record a player position for a frame."""
        if track_id not in self.player_tracks:
            self.player_tracks[track_id] = []
        self.player_tracks[track_id].append({
            "frame": frame_idx,
            "bbox": list(bbox),
            "position": list(position),
            "team_id": team_id,
            "velocity": round(velocity, 2),
        })

    def add_ball_position(self, frame_idx: int, position: tuple):
        """Record ball position."""
        self.ball_positions.append({
            "frame": frame_idx,
            "position": list(position),
        })

    def add_possession(self, frame_idx: int, team_id: int, player_id: int):
        """Record possession at a frame."""
        self.possession_log.append({
            "frame": frame_idx,
            "team_id": team_id,
            "player_id": player_id,
        })

    def compute_statistics(self) -> dict:
        """Compute aggregate match statistics."""
        stats = {
            "total_events": len(self.events),
            "event_counts": {},
            "total_players_tracked": len(self.player_tracks),
            "total_ball_detections": len(self.ball_positions),
        }

        # Event counts by type
        for evt in self.events:
            etype = evt.get("type", "unknown")
            stats["event_counts"][etype] = stats["event_counts"].get(etype, 0) + 1

        # Possession statistics
        if self.possession_log:
            team_frames = {}
            for p in self.possession_log:
                tid = p["team_id"]
                team_frames[tid] = team_frames.get(tid, 0) + 1
            total = sum(team_frames.values())
            stats["possession"] = {
                f"team_{tid}": round(count / total * 100, 1)
                for tid, count in team_frames.items()
            }

        # Player statistics
        player_stats = {}
        for pid, positions in self.player_tracks.items():
            if len(positions) < 2:
                continue
            total_distance = 0
            for i in range(1, len(positions)):
                p1 = positions[i - 1]["position"]
                p2 = positions[i]["position"]
                total_distance += ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

            avg_vel = sum(p["velocity"] for p in positions) / len(positions)

            from collections import Counter
            team_votes = [p["team_id"] for p in positions if p["team_id"] >= 0]
            if team_votes:
                majority_team = Counter(team_votes).most_common(1)[0][0]
            else:
                majority_team = -1

            player_stats[str(pid)] = {
                "team_id": majority_team,
                "frames_tracked": len(positions),
                "total_distance_px": round(float(total_distance), 1),
                "avg_velocity": round(float(avg_vel), 2),
            }
        stats["player_statistics"] = player_stats

        # v4: Freeze frame statistics
        events_with_freeze_frames = sum(1 for evt in self.events if "freeze_frame" in evt and evt["freeze_frame"])
        stats["events_with_freeze_frames"] = events_with_freeze_frames
        stats["freeze_frame_coverage"] = round(
            (events_with_freeze_frames / len(self.events) * 100) if self.events else 0, 1
        )

        return stats

    def export_json(self, filename: str = "match_analysis.json") -> str:
        """Export complete analysis to JSON file (v4: includes freeze frames)."""
        filepath = os.path.join(self.output_dir, filename)

        data = {
            "metadata": self.match_metadata,
            "statistics": self.compute_statistics(),
            "events": self.events,  # v4: Events now include freeze_frame field
            "ball_positions": self.ball_positions,
            "player_tracks": {
                str(k): v for k, v in self.player_tracks.items()
            },
            "possession_log": self.possession_log,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=_NumpySafeEncoder)

        logger.info(f"Exported match analysis to {filepath}")
        return filepath

    def export_events_json(self, filename: str = "events.json") -> str:
        """Export events to dedicated JSON format."""
        filepath = os.path.join(self.output_dir, filename)

        event_summary = {}
        for event in self.events:
            event_type = event.get("type", "unknown")
            event_summary[event_type] = event_summary.get(event_type, 0) + 1

        payload = {
            "metadata": {
                "analysis_timestamp": self.match_metadata.get("analysis_timestamp"),
                "input_video": self.match_metadata.get("input_video"),
                "total_events": len(self.events),
            },
            "event_summary": event_summary,
            "events": self.events,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, cls=_NumpySafeEncoder)

        logger.info(f"Exported {len(self.events)} events to {filepath}")
        return filepath
    
    def export_freeze_frames_json(self, filename: str = "freeze_frames.json") -> str:
        """
        Export freeze frames separately as dedicated JSON file.
        
        v4: New export function for detailed freeze frame analysis.
        """
        filepath = os.path.join(self.output_dir, filename)
        
        freeze_frames = []
        for evt in self.events:
            if "freeze_frame" in evt and evt["freeze_frame"]:
                freeze_frame_data = {
                    "event_type": evt.get("type"),
                    "frame": evt.get("frame"),
                    "timestamp": evt.get("timestamp"),
                    "player_id": evt.get("player_id"),
                    "team_id": evt.get("team_id"),
                    "position": evt.get("position"),
                    "freeze_frame": evt["freeze_frame"]
                }
                freeze_frames.append(freeze_frame_data)
        
        if freeze_frames:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(freeze_frames, f, indent=2, cls=_NumpySafeEncoder)
            
            logger.info(f"Exported {len(freeze_frames)} freeze frames to {filepath}")
        else:
            logger.info("No freeze frames to export")
        
        return filepath

    def export_player_trajectories_csv(self, filename: str = "trajectories.csv") -> str:
        """Export player trajectories to CSV."""
        filepath = os.path.join(self.output_dir, filename)

        rows = []
        for pid, positions in self.player_tracks.items():
            for pos in positions:
                rows.append({
                    "player_id": pid,
                    "frame": pos["frame"],
                    "x": pos["position"][0],
                    "y": pos["position"][1],
                    "team_id": pos["team_id"],
                    "velocity": pos["velocity"],
                })

        if rows:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

        logger.info(f"Exported trajectories to {filepath}")
        return filepath
    
    def export_event_summary_report(self, filename: str = "event_summary.txt") -> str:
        """
        Export human-readable event summary report.
        
        v4: Enhanced with freeze frame statistics.
        """
        filepath = os.path.join(self.output_dir, filename)
        
        stats = self.compute_statistics()
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("="*60 + "\n")
            f.write("MATCH ANALYSIS EVENT SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Metadata
            f.write("Match Information:\n")
            f.write("-"*60 + "\n")
            for key, value in self.match_metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            # Event statistics
            f.write("Event Statistics:\n")
            f.write("-"*60 + "\n")
            f.write(f"  Total Events: {stats['total_events']}\n")
            f.write(f"  Events with Freeze Frames: {stats.get('events_with_freeze_frames', 0)}\n")
            f.write(f"  Freeze Frame Coverage: {stats.get('freeze_frame_coverage', 0)}%\n")
            f.write("\n")
            
            # Event breakdown
            f.write("Event Breakdown:\n")
            f.write("-"*60 + "\n")
            event_counts = stats.get("event_counts", {})
            for event_type in sorted(event_counts.keys()):
                count = event_counts[event_type]
                f.write(f"  {event_type}: {count}\n")
            f.write("\n")
            
            # Possession
            if "possession" in stats:
                f.write("Possession Statistics:\n")
                f.write("-"*60 + "\n")
                for team, pct in stats["possession"].items():
                    f.write(f"  {team}: {pct}%\n")
                f.write("\n")
            
            # Player statistics summary
            f.write("Player Statistics:\n")
            f.write("-"*60 + "\n")
            f.write(f"  Total Players Tracked: {stats['total_players_tracked']}\n")
            f.write(f"  Total Ball Detections: {stats['total_ball_detections']}\n")
            f.write("\n")
        
        logger.info(f"Exported event summary to {filepath}")
        return filepath
    
    def export_all(self, base_filename: str = "match_analysis"):
        """
        Export all data formats.
        
        v4: Includes new freeze frame and summary exports.
        """
        exports = {
            "json": self.export_json(f"{base_filename}.json"),
            "events_json": self.export_events_json(f"{base_filename}_events.json"),
            "trajectories_csv": self.export_player_trajectories_csv(f"{base_filename}_trajectories.csv"),
            "freeze_frames_json": self.export_freeze_frames_json(f"{base_filename}_freeze_frames.json"),
            "summary_txt": self.export_event_summary_report(f"{base_filename}_summary.txt"),
        }
        
        logger.info(f"Exported all data formats to {self.output_dir}")
        return exports
