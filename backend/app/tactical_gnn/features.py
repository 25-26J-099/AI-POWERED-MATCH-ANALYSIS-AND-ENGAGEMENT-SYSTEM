from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch


def _safe_centroid(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros(2, dtype=np.float32)
    return points.mean(axis=0).astype(np.float32)


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    if len(positions) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def build_node_features(
    players: list[dict[str, Any]],
    event_location: list[float] | None,
    pitch_length: float,
    pitch_width: float,
    density_radius: float,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, list[str]]:
    missing_features: list[str] = []
    positions = np.array([player["location"] for player in players], dtype=np.float32)
    teammate_mask = np.array([1.0 if player.get("teammate", False) else 0.0 for player in players], dtype=np.float32)
    keeper_mask = np.array([1.0 if player.get("keeper", False) else 0.0 for player in players], dtype=np.float32)
    actor_mask = np.array([1.0 if player.get("actor", False) else 0.0 for player in players], dtype=np.float32)

    team_positions = positions[teammate_mask == 1.0]
    opponent_positions = positions[teammate_mask == 0.0]
    team_centroid = _safe_centroid(team_positions)
    opponent_centroid = _safe_centroid(opponent_positions)

    relative_centroids = np.where(
        teammate_mask[:, None] == 1.0,
        positions - team_centroid,
        positions - opponent_centroid,
    )
    pairwise_distances = _pairwise_distances(positions)
    if len(players) > 1:
        nearest_neighbor_distance = np.partition(pairwise_distances, 1, axis=1)[:, 1]
    else:
        nearest_neighbor_distance = np.zeros(len(players), dtype=np.float32)
    local_density = np.maximum((pairwise_distances <= density_radius).sum(axis=1) - 1, 0).astype(np.float32)

    if event_location is None:
        missing_features.extend(["distance_to_event", "angle_to_event"])
        event_xy = team_centroid if team_positions.size else positions.mean(axis=0)
    else:
        event_xy = np.array(event_location, dtype=np.float32)

    event_vectors = positions - event_xy
    distances_to_event = np.linalg.norm(event_vectors, axis=1).astype(np.float32)
    angles_to_event = np.arctan2(event_vectors[:, 1], event_vectors[:, 0]).astype(np.float32) / math.pi
    pitch_diagonal = math.sqrt((pitch_length ** 2) + (pitch_width ** 2))

    feature_matrix = np.column_stack(
        [
            positions[:, 0] / pitch_length,
            positions[:, 1] / pitch_width,
            teammate_mask,
            keeper_mask,
            actor_mask,
            distances_to_event / pitch_diagonal,
            angles_to_event,
            relative_centroids[:, 0] / pitch_length,
            relative_centroids[:, 1] / pitch_width,
            nearest_neighbor_distance / pitch_diagonal,
            local_density / max(len(players) - 1, 1),
        ]
    ).astype(np.float32)
    return torch.from_numpy(feature_matrix), positions, teammate_mask, missing_features


def build_edge_features(
    positions: np.ndarray,
    teammate_mask: np.ndarray,
    edge_pairs: list[tuple[int, int]],
    pitch_length: float,
    pitch_width: float,
) -> torch.Tensor:
    if not edge_pairs:
        return torch.zeros((0, 4), dtype=torch.float32)
    pitch_diagonal = math.sqrt((pitch_length ** 2) + (pitch_width ** 2))
    features: list[list[float]] = []
    for src_index, dst_index in edge_pairs:
        delta_x = float(positions[dst_index, 0] - positions[src_index, 0])
        delta_y = float(positions[dst_index, 1] - positions[src_index, 1])
        distance = math.sqrt((delta_x ** 2) + (delta_y ** 2))
        same_team = 1.0 if teammate_mask[src_index] == teammate_mask[dst_index] else 0.0
        features.append(
            [
                distance / pitch_diagonal,
                delta_x / pitch_length,
                delta_y / pitch_width,
                same_team,
            ]
        )
    return torch.tensor(features, dtype=torch.float32)
