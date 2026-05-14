"""Train player style embedding models (scaler + autoencoder + KMeans cluster).

The style pipeline in ``app/analytics/embeddings.py`` expects three artefacts:

  1. style_scaler.pkl        — sklearn StandardScaler over player-stats features
  2. style_autoencoder.pth   — PyTorch autoencoder (input_dim → 8-d embedding)
  3. style_cluster_model.pkl — sklearn KMeans (k=5 clusters over embeddings)

Input features (from embeddings.py):
  xg, xt, vaep, touches, passes, pass_accuracy,
  progressive_passes, progressive_carries,
  shots, pressures, recoveries, tackles, interceptions

Outputs
-------
  model_cache/style_scaler.pkl
  model_cache/style_autoencoder.pth
  model_cache/style_cluster_model.pkl
  metrics/style_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

FEATURE_NAMES = [
    "xg", "xt", "vaep", "touches", "passes", "pass_accuracy",
    "progressive_passes", "progressive_carries",
    "shots", "pressures", "recoveries", "tackles", "interceptions",
]
INPUT_DIM = len(FEATURE_NAMES)
EMBEDDING_DIM = 8
N_CLUSTERS = 5


# ── Autoencoder (mirrors the architecture in model_loader.py) ─────────────────

def _build_autoencoder(input_dim: int, embedding_dim: int = 8):
    """Build the same autoencoder architecture expected by embeddings.py."""
    import torch
    import torch.nn as nn

    class StyleAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, embedding_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 32), nn.ReLU(),
                nn.Linear(32, 64), nn.ReLU(),
                nn.Linear(64, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return StyleAutoencoder()


def _train_autoencoder(X_scaled: np.ndarray, embedding_dim: int, epochs: int, seed: int) -> object:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = _build_autoencoder(X_scaled.shape[1], embedding_dim).to(device)
        tensor = torch.tensor(X_scaled, dtype=torch.float32)
        loader = DataLoader(TensorDataset(tensor), batch_size=64, shuffle=True)

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimiser.zero_grad()
                loss = criterion(model(batch), batch)
                loss.backward()
                optimiser.step()
                total_loss += loss.item() * len(batch)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"  epoch {epoch + 1}/{epochs}  loss={total_loss / len(X_scaled):.6f}")

        return model.cpu()
    except ImportError:
        print("[train_style] PyTorch not available — autoencoder will be None (PCA fallback used at runtime)")
        return None


def _synthetic_dataset(n: int = 5_000, seed: int = 42) -> np.ndarray:
    """Generate synthetic per-player-per-match statistics."""
    rng = np.random.default_rng(seed)

    # 5 latent player archetypes (attacker, winger, midfielder, defender, goalkeeper)
    archetype_means = np.array([
        [0.15, 0.12, 0.05, 35, 20, 78, 4, 3, 5, 6, 5, 2, 1],    # attacker
        [0.08, 0.10, 0.03, 30, 25, 80, 6, 8, 2, 10, 6, 3, 2],   # winger
        [0.03, 0.05, 0.02, 40, 40, 88, 8, 6, 0, 15, 10, 6, 5],  # midfielder
        [0.01, 0.02, 0.01, 30, 30, 85, 3, 2, 0, 12, 8, 12, 10], # defender
        [0.00, 0.00, 0.00, 20, 15, 90, 0, 0, 0, 5, 10, 5, 3],   # goalkeeper
    ], dtype=float)

    labels = rng.choice(5, size=n)
    X = np.zeros((n, INPUT_DIM))
    for i, archetype in enumerate(labels):
        means = archetype_means[archetype]
        noise = rng.normal(0, 0.15, size=INPUT_DIM) * means
        X[i] = np.clip(means + noise, 0.0, None)

    return X


def _mlflow_log(params: dict, metrics: dict, artifacts: list[Path]) -> None:
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("style-model-training")
        with mlflow.start_run(run_name="style-autoencoder-kmeans"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for p in artifacts:
                if p.exists():
                    mlflow.log_artifact(str(p))
        print("[train_style] MLflow run logged successfully.")
    except Exception as exc:
        print(f"[train_style] MLflow logging skipped ({exc}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train player style embedding models")
    parser.add_argument("--dataset", default="data/player_style_dataset.csv")
    parser.add_argument("--output-dir", default="model_cache")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--n-samples", type=int, default=5_000)
    parser.add_argument("--n-clusters", type=int, default=N_CLUSTERS)
    parser.add_argument("--embedding-dim", type=int, default=EMBEDDING_DIM)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    dataset_path = Path(args.dataset)
    X_raw: np.ndarray | None = None
    try:
        import pandas as pd
        df = pd.read_csv(dataset_path)
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        if not missing:
            X_raw = df[FEATURE_NAMES].to_numpy(dtype=float)
            print(f"[train_style] Loaded {len(X_raw):,} samples from {dataset_path}")
    except Exception:
        pass

    if X_raw is None:
        print(f"[train_style] Generating {args.n_samples:,} synthetic player-stats samples…")
        X_raw = _synthetic_dataset(n=args.n_samples, seed=args.seed)

    # 1. Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print(f"[train_style] Scaled {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")

    # 2. Autoencoder
    print(f"[train_style] Training autoencoder ({args.epochs} epochs)…")
    autoencoder = _train_autoencoder(X_scaled, args.embedding_dim, args.epochs, args.seed)

    # Compute embeddings for KMeans
    if autoencoder is not None:
        import torch
        autoencoder.eval()
        with torch.no_grad():
            embeddings = autoencoder.encoder(
                torch.tensor(X_scaled, dtype=torch.float32)
            ).numpy()
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=args.embedding_dim, random_state=args.seed)
        embeddings = pca.fit_transform(X_scaled)

    # 3. KMeans
    print(f"[train_style] Fitting KMeans (k={args.n_clusters})…")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=args.seed, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    sil = float(silhouette_score(embeddings, cluster_labels))

    metrics = {
        "silhouette_score": round(sil, 4),
        "n_clusters": args.n_clusters,
        "n_samples": len(X_raw),
        "embedding_dim": args.embedding_dim,
        "autoencoder_available": autoencoder is not None,
    }
    params = {
        "n_clusters": args.n_clusters,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "seed": args.seed,
    }

    scaler_path = output_dir / "style_scaler.pkl"
    autoencoder_path = output_dir / "style_autoencoder.pth"
    kmeans_path = output_dir / "style_cluster_model.pkl"
    metrics_path = metrics_dir / "style_metrics.json"

    joblib.dump(scaler, scaler_path)
    if autoencoder is not None:
        import torch
        torch.save(autoencoder.state_dict(), autoencoder_path)
    joblib.dump(kmeans, kmeans_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train_style] silhouette={sil:.4f}")
    print(f"[train_style] Saved → {scaler_path}, {autoencoder_path}, {kmeans_path}")
    print(f"[train_style] Saved metrics → {metrics_path}")

    _mlflow_log(
        params=params,
        metrics=metrics,
        artifacts=[scaler_path, kmeans_path, metrics_path],
    )


if __name__ == "__main__":
    main()
