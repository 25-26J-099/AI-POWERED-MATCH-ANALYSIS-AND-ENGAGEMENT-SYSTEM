from __future__ import annotations

import copy
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.config.settings import settings
from app.tactical_gnn.schemas import TacticalGNNConfig

LOGGER = logging.getLogger(__name__)


def get_event_location(event_data: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(event_data, dict):
        return None
    loc = event_data.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        try:
            return [float(loc[0]), float(loc[1])]
        except (TypeError, ValueError):
            return None
    return None


def infer_attacking_direction_right(
    freeze_frame_data: dict[str, Any] | None,
    pitch_length: float = 120.0,
) -> bool:
    if not freeze_frame_data or "freeze_frame" not in freeze_frame_data:
        return True
    teammates_gk_x: list[float] = []
    opponents_gk_x: list[float] = []
    for player in freeze_frame_data["freeze_frame"]:
        if not isinstance(player, dict):
            continue
        loc = player.get("location")
        if not isinstance(loc, (list, tuple)) or len(loc) < 2:
            continue
        try:
            x_value = float(loc[0])
        except (TypeError, ValueError):
            continue
        teammate = bool(player.get("teammate", False))
        keeper = bool(player.get("keeper", False))
        if keeper and teammate:
            teammates_gk_x.append(x_value)
        elif keeper:
            opponents_gk_x.append(x_value)
    if teammates_gk_x and opponents_gk_x:
        return float(np.mean(opponents_gk_x)) > float(np.mean(teammates_gk_x))
    if opponents_gk_x:
        return float(np.mean(opponents_gk_x)) > (pitch_length / 2.0)
    if teammates_gk_x:
        return float(np.mean(teammates_gk_x)) < (pitch_length / 2.0)
    return True


def normalize_x(x_value: float, attacking_right: bool = True, pitch_length: float = 120.0) -> float:
    return float(x_value) if attacking_right else float(pitch_length - float(x_value))


def normalize_location(
    location: list[float] | tuple[float, float] | None,
    attacking_right: bool = True,
    pitch_length: float = 120.0,
) -> list[float] | None:
    if not location or len(location) < 2:
        return None
    try:
        x_value = normalize_x(float(location[0]), attacking_right=attacking_right, pitch_length=pitch_length)
        y_value = float(location[1])
    except (TypeError, ValueError):
        return None
    return [x_value, y_value]


def extract_freeze_frame_players(freeze_frame_data: Any) -> list[dict[str, Any]]:
    if freeze_frame_data is None:
        return []
    if isinstance(freeze_frame_data, list):
        players = freeze_frame_data
    elif isinstance(freeze_frame_data, dict):
        if "freeze_frame" in freeze_frame_data:
            players = freeze_frame_data.get("freeze_frame") or []
        elif "players" in freeze_frame_data:
            players = freeze_frame_data.get("players") or []
        else:
            players = []
    else:
        players = []
    return [player for player in players if isinstance(player, dict)]


def normalize_freeze_frame(
    freeze_frame_data: dict[str, Any] | None,
    attacking_right: bool = True,
    pitch_length: float = 120.0,
) -> dict[str, Any] | None:
    if not freeze_frame_data or "freeze_frame" not in freeze_frame_data:
        return None
    freeze_frame_copy = copy.deepcopy(freeze_frame_data)
    normalized_players: list[dict[str, Any]] = []
    for player in extract_freeze_frame_players(freeze_frame_copy):
        normalized_location = normalize_location(
            player.get("location"),
            attacking_right=attacking_right,
            pitch_length=pitch_length,
        )
        if normalized_location is None:
            continue
        player["location"] = normalized_location
        normalized_players.append(player)
    freeze_frame_copy["freeze_frame"] = normalized_players
    return freeze_frame_copy


def normalize_event_data(
    event_data: dict[str, Any] | None,
    attacking_right: bool = True,
    pitch_length: float = 120.0,
) -> dict[str, Any] | None:
    if not isinstance(event_data, dict):
        return None
    event_copy = copy.deepcopy(event_data)
    location = get_event_location(event_copy)
    if location is not None:
        event_copy["location"] = normalize_location(
            location,
            attacking_right=attacking_right,
            pitch_length=pitch_length,
        )
    return event_copy


def normalize_snapshot(
    event_data: dict[str, Any] | None,
    freeze_frame_data: dict[str, Any] | None,
    pitch_length: float = 120.0,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool, bool]:
    explicit_direction = None
    if isinstance(event_data, dict) and isinstance(event_data.get("attacking_right"), bool):
        explicit_direction = event_data["attacking_right"]
    elif isinstance(freeze_frame_data, dict) and isinstance(freeze_frame_data.get("attacking_right"), bool):
        explicit_direction = freeze_frame_data["attacking_right"]
    attacking_right = (
        bool(explicit_direction)
        if explicit_direction is not None
        else infer_attacking_direction_right(freeze_frame_data, pitch_length=pitch_length)
    )
    return (
        normalize_event_data(event_data, attacking_right=attacking_right, pitch_length=pitch_length),
        normalize_freeze_frame(freeze_frame_data, attacking_right=attacking_right, pitch_length=pitch_length),
        attacking_right,
        True,
    )


def resolve_model_path(model_path: str | None, fallback: str | None = None) -> str | None:
    candidate = model_path or fallback
    if not candidate:
        return None
    return str(Path(candidate).expanduser())


def resolve_device(requested_device: str | None = None) -> str:
    if getattr(settings, "FORCE_CPU", False):
        return "cpu"
    requested = str(requested_device or "cpu").lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return requested
    return "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu"


def config_from_settings() -> TacticalGNNConfig:
    return TacticalGNNConfig(
        enabled=getattr(settings, "ENABLE_GNN_TACTICAL_ANALYSIS", True),
        model_path=getattr(settings, "GNN_MODEL_PATH", None),
        device=getattr(settings, "GNN_DEVICE", "cpu"),
        edge_strategy=getattr(settings, "GNN_EDGE_STRATEGY", "knn"),
        k_neighbors=getattr(settings, "GNN_K_NEIGHBORS", 4),
        radius=getattr(settings, "GNN_RADIUS", 18.0),
        confidence_threshold=getattr(settings, "GNN_CONFIDENCE_THRESHOLD", 0.4),
        use_heuristic_fallback=getattr(settings, "GNN_USE_HEURISTIC_FALLBACK", True),
    )


def ensure_directory(path_value: str | os.PathLike[str]) -> Path:
    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_log_once(cache: set[str], key: str, message: str) -> None:
    if key in cache:
        return
    cache.add(key)
    LOGGER.warning(message)
