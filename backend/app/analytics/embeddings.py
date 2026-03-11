"""Player style embeddings aligned with the training notebooks."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.gpu_config import get_device
from app.services.model_loader import (
    get_style_scaler_feature_count,
    get_style_scaler_feature_names,
    load_style_models,
)

BASE_FEATURES = [
    "xT",
    "x",
    "y",
    "nearest_def",
    "teammates_ahead",
    "defenders_ahead",
    "xG",
    "vaep",
    "touches",
]
RATE_COLUMNS = ["Pass", "Carry", "Shot", "Pressure", "Miscontrol", "Ball Receipt*"]
FALLBACK_FEATURE_ORDER = [
    "xT",
    "x",
    "y",
    "nearest_def",
    "teammates_ahead",
    "defenders_ahead",
    "50/50",
    "Ball Recovery",
    "Block",
    "Clearance",
    "Dispossessed",
    "Dribble",
    "Duel",
    "Foul Committed",
    "Foul Won",
    "Interception",
    "Offside",
    "Shield",
    "xG",
    "vaep",
    "touches",
    "Pass_rate",
    "Carry_rate",
    "Shot_rate",
    "Pressure_rate",
    "Miscontrol_rate",
    "Ball Receipt*_rate",
    "Own Goal Against",
    "Error",
    "Dribbled Past",
    "Goal Keeper",
]


def _freeze_frame_players(event: dict) -> list[dict]:
    freeze_frame_raw = event.get("freeze_frame", [])
    if isinstance(freeze_frame_raw, dict):
        return freeze_frame_raw.get("players", [])
    return freeze_frame_raw if isinstance(freeze_frame_raw, list) else []


def _event_type_name(event: dict) -> str:
    raw_type = event.get("type", "")
    if isinstance(raw_type, dict):
        return str(raw_type.get("name", ""))
    return str(raw_type)


def _player_row(player_payload: Dict) -> dict:
    player_events = player_payload["player_events"]
    valid_events = [
        event for event in player_events if event.get("location") and len(event["location"]) >= 2
    ]

    xs = [float(event["location"][0]) for event in valid_events]
    ys = [float(event["location"][1]) for event in valid_events]

    nearest_def = []
    teammates_ahead = []
    defenders_ahead = []

    for event in valid_events:
        event_x, event_y = float(event["location"][0]), float(event["location"][1])
        nearest_defender = 999.0
        teammates_ahead_count = 0
        defenders_ahead_count = 0

        for player in _freeze_frame_players(event):
            player_location = player.get("location")
            if not player_location or len(player_location) < 2:
                continue
            px, py = float(player_location[0]), float(player_location[1])
            distance = np.linalg.norm(np.array([px, py]) - np.array([event_x, event_y]))

            if player.get("teammate", True):
                if px > event_x:
                    teammates_ahead_count += 1
            else:
                nearest_defender = min(nearest_defender, distance)
                if px > event_x:
                    defenders_ahead_count += 1

        if nearest_defender == 999.0:
            nearest_defender = 0.0
        nearest_def.append(nearest_defender)
        teammates_ahead.append(teammates_ahead_count)
        defenders_ahead.append(defenders_ahead_count)

    type_counts = Counter(_event_type_name(event) for event in valid_events)
    touches = len(valid_events)

    row = {
        "xT": float(player_payload["xt"]),
        "x": float(np.mean(xs)) if xs else 0.0,
        "y": float(np.mean(ys)) if ys else 0.0,
        "nearest_def": float(np.mean(nearest_def)) if nearest_def else 0.0,
        "teammates_ahead": float(sum(teammates_ahead)) / max(len(valid_events), 1),
        "defenders_ahead": float(sum(defenders_ahead)) / max(len(valid_events), 1),
        "xG": float(player_payload["xg"]),
        "vaep": float(player_payload["vaep"]),
        "touches": float(touches),
    }

    for event_name, count in type_counts.items():
        row[event_name] = float(count)

    for column in RATE_COLUMNS:
        row[f"{column}_rate"] = float(row.get(column, 0.0) / (row["touches"] + 1.0))

    for column in RATE_COLUMNS:
        row.pop(column, None)

    row["player_id"] = player_payload["player_id"]
    return row


def _resolve_feature_order(scaler) -> list[str]:
    if scaler is not None:
        feature_names = get_style_scaler_feature_names()
        if feature_names and not feature_names[0].startswith("feature_"):
            return feature_names
        n_features = get_style_scaler_feature_count()
        if n_features == len(FALLBACK_FEATURE_ORDER):
            return FALLBACK_FEATURE_ORDER
    return FALLBACK_FEATURE_ORDER


def _build_feature_matrix(players_data: List[Dict], scaler) -> tuple[list[int], np.ndarray]:
    rows = [_player_row(player) for player in players_data]
    if not rows:
        return [], np.empty((0, 0))

    feature_order = _resolve_feature_order(scaler)
    valid_player_ids = [int(row.pop("player_id")) for row in rows]
    frame = pd.DataFrame(rows).fillna(0.0)

    for column in feature_order:
        if column not in frame.columns:
            frame[column] = 0.0

    frame = frame[feature_order]
    frame = frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if "xG" in frame.columns:
        frame["xG"] = np.log1p(frame["xG"])
    if "touches" in frame.columns:
        frame["touches"] = np.log1p(frame["touches"])

    if scaler is not None and hasattr(scaler, "n_features_in_"):
        expected = int(scaler.n_features_in_)
        if frame.shape[1] != expected:
            raise ValueError(f"Style feature mismatch: built {frame.shape[1]} features but scaler expects {expected}")

    return valid_player_ids, frame.to_numpy(dtype=np.float32)


def compute_embeddings(players_data: List[Dict]) -> List[Dict]:
    scaler, autoencoder, kmeans = load_style_models()
    valid_players, X = _build_feature_matrix(players_data, scaler)
    if not valid_players:
        return []

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    if autoencoder is not None:
        import torch

        device = get_device()
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            embeddings = autoencoder.encoder(X_tensor).cpu().numpy()
    else:
        from sklearn.decomposition import PCA

        n_components = min(8, X_scaled.shape[1], X_scaled.shape[0])
        embeddings = PCA(n_components=n_components).fit_transform(X_scaled)

    if kmeans is not None:
        clusters = kmeans.predict(embeddings)
    else:
        from sklearn.cluster import KMeans

        n_clusters = min(4, len(valid_players))
        clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(embeddings)

    umap_coords = _reduce_umap(embeddings)
    tsne_coords = _reduce_tsne(embeddings)

    return [
        {
            "player_id": player_id,
            "embedding": embeddings[index].tolist(),
            "umap_x": float(umap_coords[index, 0]) if umap_coords is not None else 0.0,
            "umap_y": float(umap_coords[index, 1]) if umap_coords is not None else 0.0,
            "tsne_x": float(tsne_coords[index, 0]) if tsne_coords is not None else 0.0,
            "tsne_y": float(tsne_coords[index, 1]) if tsne_coords is not None else 0.0,
            "cluster": int(clusters[index]),
        }
        for index, player_id in enumerate(valid_players)
    ]


def _reduce_umap(embeddings: np.ndarray) -> Optional[np.ndarray]:
    try:
        import umap

        n_neighbors = min(15, len(embeddings) - 1)
        if n_neighbors < 2:
            return embeddings[:, :2] if embeddings.shape[1] >= 2 else None
        return umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(embeddings)
    except Exception:
        return embeddings[:, :2] if embeddings.shape[1] >= 2 else None


def _reduce_tsne(embeddings: np.ndarray) -> Optional[np.ndarray]:
    try:
        from sklearn.manifold import TSNE

        perplexity = min(30, len(embeddings) - 1)
        if perplexity < 2:
            return embeddings[:, :2] if embeddings.shape[1] >= 2 else None
        return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
    except Exception:
        return embeddings[:, :2] if embeddings.shape[1] >= 2 else None
