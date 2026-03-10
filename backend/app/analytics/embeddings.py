"""Player style embeddings using Scaler + StyleAutoencoder + KMeans pipeline."""

import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from app.services.model_loader import load_style_models
from app.config.gpu_config import get_device


# ── The 26 features used by the embedding pipeline ───────────────────────
EMBEDDING_FEATURES = [
    "xT", "x", "y", "nearest_def", "teammates_ahead", "defenders_ahead",
    "50/50", "Ball Recovery", "Block", "Clearance", "Dribble", "Duel",
    "Foul Committed", "Foul Won", "Interception", "Offside", "Shield",
    "xG", "vaep", "touches",
    "Pass_rate", "Carry_rate", "Shot_rate", "Pressure_rate",
    "Miscontrol_rate", "Ball Receipt*_rate",
]


def _extract_player_features(
    player_events: List[dict],
    player_xt: float,
    player_xg: float,
    player_vaep: float,
    player_touches: int,
) -> Optional[np.ndarray]:
    """Build the 26-feature vector for one player.

    Args:
        player_events: Raw events for this player.
        player_xt, player_xg, player_vaep, player_touches: Pre-computed stats.

    Returns:
        Feature vector (26,) or None if insufficient data.
    """
    if not player_events or player_touches < 5:
        return None

    # Spatial features (mean location)
    xs = [e["location"][0] for e in player_events if e.get("location") and len(e["location"]) >= 2]
    ys = [e["location"][1] for e in player_events if e.get("location") and len(e["location"]) >= 2]
    avg_x = np.mean(xs) if xs else 60.0
    avg_y = np.mean(ys) if ys else 40.0

    # Freeze frame based features (averaged)
    nearest_defs = []
    teammates_ahead_counts = []
    defenders_ahead_counts = []

    for e in player_events:
        ff_raw = e.get("freeze_frame", [])
        ff = ff_raw.get("players", []) if isinstance(ff_raw, dict) else ff_raw
        if not ff:
            continue
        loc = e.get("location", [0, 0])
        if len(loc) < 2:
            continue
        ex, ey = loc[0], loc[1]

        defenders = [p for p in ff if not p.get("teammate", True) and not p.get("keeper", False)]
        teammates = [p for p in ff if p.get("teammate", True) and not p.get("actor", False)]

        if defenders:
            dds = [
                math.sqrt((p["location"][0] - ex) ** 2 + (p["location"][1] - ey) ** 2)
                for p in defenders if "location" in p and len(p["location"]) >= 2
            ]
            if dds:
                nearest_defs.append(min(dds))

        # Teammates ahead (closer to opponent goal, x > player x)
        ta = sum(1 for p in teammates if "location" in p and len(p["location"]) >= 2 and p["location"][0] > ex)
        teammates_ahead_counts.append(ta)

        # Defenders ahead
        da = sum(1 for p in defenders if "location" in p and len(p["location"]) >= 2 and p["location"][0] > ex)
        defenders_ahead_counts.append(da)

    nearest_def = np.mean(nearest_defs) if nearest_defs else 15.0
    teammates_ahead = np.mean(teammates_ahead_counts) if teammates_ahead_counts else 3.0
    defenders_ahead = np.mean(defenders_ahead_counts) if defenders_ahead_counts else 4.0

    # Event type counts
    def _type(e):
        t = e.get("type", "")
        return t.get("name", "") if isinstance(t, dict) else str(t)

    type_counts = {}
    total_events = len(player_events)
    for e in player_events:
        t = _type(e)
        type_counts[t] = type_counts.get(t, 0) + 1

    # Action counts for specific types
    fifty_fifty = type_counts.get("50/50", 0)
    ball_recovery = type_counts.get("Ball Recovery", 0)
    block = type_counts.get("Block", 0)
    clearance = type_counts.get("Clearance", 0)
    dribble = type_counts.get("Dribble", 0)
    duel = type_counts.get("Duel", 0)
    foul_committed = type_counts.get("Foul Committed", 0)
    foul_won = type_counts.get("Foul Won", 0)
    interception = type_counts.get("Interception", 0)
    offside = type_counts.get("Offside", 0)
    shield = type_counts.get("Shield", 0)

    # Rates (per event)
    pass_rate = type_counts.get("Pass", 0) / max(total_events, 1)
    carry_rate = type_counts.get("Carry", 0) / max(total_events, 1)
    shot_rate = type_counts.get("Shot", 0) / max(total_events, 1)
    pressure_rate = type_counts.get("Pressure", 0) / max(total_events, 1)
    miscontrol_rate = type_counts.get("Miscontrol", 0) / max(total_events, 1)
    ball_receipt_rate = type_counts.get("Ball Receipt*", 0) / max(total_events, 1)

    # Log-transform xG and touches
    log_xg = math.log(player_xg + 1e-6)
    log_touches = math.log(player_touches + 1)

    features = np.array([
        player_xt, avg_x, avg_y, nearest_def, teammates_ahead, defenders_ahead,
        fifty_fifty, ball_recovery, block, clearance, dribble, duel,
        foul_committed, foul_won, interception, offside, shield,
        log_xg, player_vaep, log_touches,
        pass_rate, carry_rate, shot_rate, pressure_rate,
        miscontrol_rate, ball_receipt_rate,
    ], dtype=np.float32)

    return features


def compute_embeddings(
    players_data: List[Dict],
) -> List[Dict]:
    """Compute style embeddings for all players using the 3-model pipeline.

    Args:
        players_data: List of dicts with keys:
            player_id, player_events, xt, xg, vaep, touches

    Returns:
        List of dicts with: player_id, embedding, umap_x, umap_y, tsne_x, tsne_y, cluster
    """
    # Extract features for all players
    valid_players = []
    feature_matrix = []

    for pd in players_data:
        features = _extract_player_features(
            pd["player_events"], pd["xt"], pd["xg"], pd["vaep"], pd["touches"]
        )
        if features is not None:
            valid_players.append(pd["player_id"])
            feature_matrix.append(features)

    if not feature_matrix:
        return []

    X = np.vstack(feature_matrix)

    # Load the 3 models
    scaler, autoencoder, kmeans = load_style_models()

    # Step 1: Scale
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # Step 2: Encode via autoencoder
    if autoencoder is not None:
        import torch
        device = get_device()
        autoencoder = autoencoder.to(device)
        autoencoder.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            embeddings = autoencoder.encode(X_tensor).cpu().numpy()
    else:
        # Fallback: use PCA-like dimensionality reduction
        from sklearn.decomposition import PCA
        n_components = min(8, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(X_scaled)

    # Step 3: Cluster
    if kmeans is not None:
        clusters = kmeans.predict(embeddings)
    else:
        from sklearn.cluster import KMeans as KM
        n_clusters = min(4, len(valid_players))
        km = KM(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = km.fit_predict(embeddings)

    # Step 4: Dimensionality reduction for visualization
    umap_coords = _reduce_umap(embeddings)
    tsne_coords = _reduce_tsne(embeddings)

    results = []
    for i, pid in enumerate(valid_players):
        results.append({
            "player_id": pid,
            "embedding": embeddings[i].tolist(),
            "umap_x": float(umap_coords[i, 0]) if umap_coords is not None else 0.0,
            "umap_y": float(umap_coords[i, 1]) if umap_coords is not None else 0.0,
            "tsne_x": float(tsne_coords[i, 0]) if tsne_coords is not None else 0.0,
            "tsne_y": float(tsne_coords[i, 1]) if tsne_coords is not None else 0.0,
            "cluster": int(clusters[i]),
        })

    return results


def _reduce_umap(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Reduce embeddings to 2D using UMAP."""
    try:
        import umap
        n_neighbors = min(15, len(embeddings) - 1)
        if n_neighbors < 2:
            return embeddings[:, :2] if embeddings.shape[1] >= 2 else None
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        return reducer.fit_transform(embeddings)
    except ImportError:
        # Fallback to first 2 components
        return embeddings[:, :2] if embeddings.shape[1] >= 2 else None


def _reduce_tsne(embeddings: np.ndarray) -> Optional[np.ndarray]:
    """Reduce embeddings to 2D using t-SNE."""
    try:
        from sklearn.manifold import TSNE
        perplexity = min(30, len(embeddings) - 1)
        if perplexity < 2:
            return embeddings[:, :2] if embeddings.shape[1] >= 2 else None
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        return tsne.fit_transform(embeddings)
    except Exception:
        return embeddings[:, :2] if embeddings.shape[1] >= 2 else None
