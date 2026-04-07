from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from app.tactical_gnn.features import build_edge_features, build_node_features
from app.tactical_gnn.schemas import GraphMetadataModel, TacticalGNNConfig
from app.tactical_gnn.utils import extract_freeze_frame_players, get_event_location, normalize_snapshot


@dataclass(slots=True)
class FreezeFrameGraph:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor
    teammate_mask: torch.Tensor
    metadata: GraphMetadataModel


def _sanitize_players(players: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    sanitized: list[dict[str, Any]] = []
    missing_features: list[str] = []
    for index, player in enumerate(players):
        location = player.get("location")
        if not isinstance(location, (list, tuple)) or len(location) < 2:
            missing_features.append(f"player_{index}_location")
            continue
        try:
            x_value = float(location[0])
            y_value = float(location[1])
        except (TypeError, ValueError):
            missing_features.append(f"player_{index}_location")
            continue
        normalized_player = dict(player)
        normalized_player["location"] = [x_value, y_value]
        normalized_player["teammate"] = bool(player.get("teammate", False))
        normalized_player["keeper"] = bool(player.get("keeper", False))
        normalized_player["actor"] = bool(player.get("actor", False))
        sanitized.append(normalized_player)
    return sanitized, missing_features


def _collect_invalid_location_features(players: list[dict[str, Any]]) -> list[str]:
    missing_features: list[str] = []
    for index, player in enumerate(players):
        location = player.get("location")
        if not isinstance(location, (list, tuple)) or len(location) < 2:
            missing_features.append(f"player_{index}_location")
            continue
        try:
            float(location[0])
            float(location[1])
        except (TypeError, ValueError):
            missing_features.append(f"player_{index}_location")
    return missing_features


def _build_edge_pairs(positions: np.ndarray, config: TacticalGNNConfig) -> list[tuple[int, int]]:
    player_count = len(positions)
    if player_count < 2:
        return []
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(distances, np.inf)
    edge_pairs: set[tuple[int, int]] = set()

    if config.edge_strategy == "radius":
        for src_index in range(player_count):
            neighbor_indices = np.where(distances[src_index] <= config.radius)[0]
            for dst_index in sorted(int(idx) for idx in neighbor_indices):
                edge_pairs.add((src_index, dst_index))
                edge_pairs.add((dst_index, src_index))
    else:
        k_value = min(config.k_neighbors, player_count - 1)
        for src_index in range(player_count):
            ordered_neighbors = np.argsort(distances[src_index], kind="stable")[:k_value]
            for dst_index in ordered_neighbors.tolist():
                edge_pairs.add((src_index, int(dst_index)))
                edge_pairs.add((int(dst_index), src_index))

    return sorted(edge_pairs)


def build_graph_from_snapshot(
    event_data: dict[str, Any] | None,
    freeze_frame_data: dict[str, Any] | None,
    config: TacticalGNNConfig | None = None,
) -> FreezeFrameGraph:
    graph_config = config or TacticalGNNConfig()
    raw_input_players = extract_freeze_frame_players(freeze_frame_data or {})
    missing_features = _collect_invalid_location_features(raw_input_players)
    normalized_event, normalized_frame, attacking_right, normalization_applied = normalize_snapshot(
        event_data,
        freeze_frame_data,
        pitch_length=graph_config.pitch_length,
    )
    raw_players = extract_freeze_frame_players(normalized_frame or {})
    players, sanitize_missing_features = _sanitize_players(raw_players)
    missing_features.extend(sanitize_missing_features)
    event_location = get_event_location(normalized_event)

    if len(players) < graph_config.min_players:
        metadata = GraphMetadataModel(
            num_nodes=len(players),
            num_edges=0,
            normalization_applied=normalization_applied,
            missing_features=missing_features,
            edge_strategy=graph_config.edge_strategy,
            attacking_right=attacking_right,
            insufficient_data=True,
            fallback_reason=f"insufficient visible players ({len(players)})",
        )
        return FreezeFrameGraph(
            x=torch.zeros((len(players), 11), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, 4), dtype=torch.float32),
            batch=torch.zeros((len(players),), dtype=torch.long),
            teammate_mask=torch.zeros((len(players),), dtype=torch.float32),
            metadata=metadata,
        )

    node_features, positions, teammate_mask, feature_missing = build_node_features(
        players=players,
        event_location=event_location,
        pitch_length=graph_config.pitch_length,
        pitch_width=graph_config.pitch_width,
        density_radius=graph_config.local_density_radius,
    )
    missing_features.extend(feature_missing)
    edge_pairs = _build_edge_pairs(positions, graph_config)
    edge_index = (
        torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        if edge_pairs
        else torch.zeros((2, 0), dtype=torch.long)
    )
    edge_attr = build_edge_features(
        positions=positions,
        teammate_mask=teammate_mask,
        edge_pairs=edge_pairs,
        pitch_length=graph_config.pitch_length,
        pitch_width=graph_config.pitch_width,
    )
    metadata = GraphMetadataModel(
        num_nodes=int(node_features.shape[0]),
        num_edges=int(edge_index.shape[1]),
        normalization_applied=normalization_applied,
        missing_features=missing_features,
        edge_strategy=graph_config.edge_strategy,
        attacking_right=attacking_right,
        insufficient_data=False,
    )
    return FreezeFrameGraph(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros((node_features.shape[0],), dtype=torch.long),
        teammate_mask=torch.from_numpy(teammate_mask.astype(np.float32)),
        metadata=metadata,
    )


def collate_graphs(graphs: list[FreezeFrameGraph]) -> FreezeFrameGraph:
    if not graphs:
        raise ValueError("at least one graph is required for collation")
    x_parts: list[torch.Tensor] = []
    edge_parts: list[torch.Tensor] = []
    edge_attr_parts: list[torch.Tensor] = []
    batch_parts: list[torch.Tensor] = []
    teammate_parts: list[torch.Tensor] = []
    offset = 0
    missing_features: list[str] = []
    for graph_index, graph in enumerate(graphs):
        x_parts.append(graph.x)
        teammate_parts.append(graph.teammate_mask)
        batch_parts.append(torch.full((graph.x.shape[0],), graph_index, dtype=torch.long))
        if graph.edge_index.numel():
            edge_parts.append(graph.edge_index + offset)
            edge_attr_parts.append(graph.edge_attr)
        offset += graph.x.shape[0]
        missing_features.extend(graph.metadata.missing_features)
    merged_edge_index = (
        torch.cat(edge_parts, dim=1) if edge_parts else torch.zeros((2, 0), dtype=torch.long)
    )
    merged_edge_attr = (
        torch.cat(edge_attr_parts, dim=0) if edge_attr_parts else torch.zeros((0, 4), dtype=torch.float32)
    )
    metadata = GraphMetadataModel(
        num_nodes=int(sum(graph.metadata.num_nodes for graph in graphs)),
        num_edges=int(sum(graph.metadata.num_edges for graph in graphs)),
        normalization_applied=all(graph.metadata.normalization_applied for graph in graphs),
        missing_features=missing_features,
        edge_strategy=graphs[0].metadata.edge_strategy,
        attacking_right=graphs[0].metadata.attacking_right,
        insufficient_data=any(graph.metadata.insufficient_data for graph in graphs),
    )
    return FreezeFrameGraph(
        x=torch.cat(x_parts, dim=0),
        edge_index=merged_edge_index,
        edge_attr=merged_edge_attr,
        batch=torch.cat(batch_parts, dim=0),
        teammate_mask=torch.cat(teammate_parts, dim=0),
        metadata=metadata,
    )
