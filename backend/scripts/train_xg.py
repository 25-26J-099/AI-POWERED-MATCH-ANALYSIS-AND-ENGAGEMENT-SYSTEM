"""Train the Expected-Goals (xG) prediction model.

Usage
-----
    python scripts/train_xg.py [--dataset data/shots_dataset.csv] [--output-dir model_cache/]

Features (9) match ``app/analytics/xg.py`` feature extraction:
  distance, angle, log_distance, distance_squared,
  angle_distance_interaction, pressure_weighted_distance,
  defender_count, nearest_defender_distance, goalkeeper_distance

Target: is_goal (binary 0/1)

Outputs
-------
  model_cache/xg_model.pkl      sklearn GradientBoostingClassifier
  metrics/xg_metrics.json       accuracy + AUC-ROC + Brier score
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

FEATURE_NAMES = [
    "distance", "angle", "log_distance", "distance_squared",
    "angle_distance_interaction", "pressure_weighted_distance",
    "defender_count", "nearest_defender_distance", "goalkeeper_distance",
]


def _synthetic_dataset(n: int = 30_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic shot data with realistic xG distributions."""
    rng = np.random.default_rng(seed)

    # Shot locations: roughly scaled to StatsBomb pitch (105 × 68 m → normalised 0-1)
    shot_x = rng.uniform(0.0, 1.0, n)   # 0=own goal, 1=opponent goal
    shot_y = rng.uniform(0.0, 1.0, n)   # 0=bottom, 1=top

    # Geometric features
    dist_m = np.sqrt((1.0 - shot_x) ** 2 + (0.5 - shot_y) ** 2) * 42.0  # ≈ metres
    angle = np.arctan2(np.abs(0.5 - shot_y), np.abs(1.0 - shot_x))

    log_dist = np.log1p(dist_m)
    dist_sq = dist_m ** 2
    ang_dist = angle * dist_m
    under_pressure = rng.choice([0, 1], size=n, p=[0.6, 0.4]).astype(float)
    pressure_w_dist = dist_m * (1.0 + 0.3 * under_pressure)

    def_count = rng.integers(0, 4, n).astype(float)
    nearest_def = dist_m * rng.uniform(0.1, 0.8, n)
    gk_dist = dist_m * rng.uniform(0.05, 0.5, n)

    X = np.column_stack([
        dist_m, angle, log_dist, dist_sq, ang_dist, pressure_w_dist,
        def_count, nearest_def, gk_dist,
    ])

    # Goal probability: physically motivated model
    log_odds = (
        2.5
        - 0.12 * dist_m
        + 0.8 * angle
        - 0.15 * def_count
        - 0.05 * dist_sq / 100
        + rng.normal(0, 0.6, n)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < prob).astype(int)
    return X, y


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import pandas as pd
        df = pd.read_csv(path)
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing or "is_goal" not in df.columns:
            print(f"[train_xg] CSV missing columns — using synthetic data")
            return None
        return df[FEATURE_NAMES].to_numpy(dtype=float), df["is_goal"].to_numpy(dtype=int)
    except Exception as exc:
        print(f"[train_xg] Could not load CSV: {exc} — using synthetic data")
        return None


def _mlflow_log(params: dict, metrics: dict, artifacts: list[Path]) -> None:
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("xg-model-training")
        with mlflow.start_run(run_name="xg-gbm"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for p in artifacts:
                if p.exists():
                    mlflow.log_artifact(str(p))
        print("[train_xg] MLflow run logged successfully.")
    except Exception as exc:
        print(f"[train_xg] MLflow logging skipped ({exc}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train xG model")
    parser.add_argument("--dataset", default="data/shots_dataset.csv")
    parser.add_argument("--output-dir", default="model_cache")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--n-samples", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    loaded = _load_csv(dataset_path)
    if loaded is None:
        print(f"[train_xg] Generating {args.n_samples:,} synthetic shot samples…")
        X, y = _synthetic_dataset(n=args.n_samples, seed=args.seed)
    else:
        X, y = loaded
        print(f"[train_xg] Loaded {len(X):,} samples from {dataset_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=args.seed,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    brier = float(brier_score_loss(y_test, y_prob))

    metrics = {
        "accuracy": round(accuracy, 4),
        "auc_roc": round(auc, 4),
        "brier_score": round(brier, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "goal_rate": round(float(y.mean()), 4),
    }
    params = {
        "model": "GradientBoostingClassifier",
        "n_estimators": 200, "max_depth": 4,
        "learning_rate": 0.05, "seed": args.seed,
    }

    model_path = output_dir / "xg_model.pkl"
    metrics_path = metrics_dir / "xg_metrics.json"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train_xg] accuracy={accuracy:.4f}  auc={auc:.4f}  brier={brier:.4f}")
    print(f"[train_xg] Saved model → {model_path}")
    print(f"[train_xg] Saved metrics → {metrics_path}")

    _mlflow_log(params=params, metrics=metrics, artifacts=[model_path, metrics_path])


if __name__ == "__main__":
    main()
