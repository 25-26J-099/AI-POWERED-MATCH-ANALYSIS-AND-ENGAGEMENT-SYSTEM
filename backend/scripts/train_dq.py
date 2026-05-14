"""Train the Decision-Quality (DQ) prediction model.

Usage
-----
    python scripts/train_dq.py [--dataset data/final_dataset.csv] [--output-dir data/]

If no dataset CSV is provided (or the file does not exist) the script generates a
synthetic training set whose feature distributions closely match the real DQ pipeline
outputs. This makes ``dvc repro`` runnable even without labelled match data.

The model is a sklearn LogisticRegression trained on the 19-feature vector produced
by ``decision_quality._extract_state`` + ``_compute_opponent_features``:

  State  (10): ball_x, ball_y, dist_to_goal, angle_to_goal, nearest_defender_dist,
               num_defenders_close, opponent_density, defensive_compactness,
               nearest_teammate_dist, defenders_ahead
  Candidate (3): target_x, target_y, distance
  Opponent  (6): cand_nearest_def_dist, cand_avg_top2_def_dist,
                 cand_num_defenders_near, cand_num_defenders_in_lane,
                 cand_min_def_dist_to_lane, cand_defenders_ahead

Target: action_success  (binary 0/1)

Outputs
-------
  data/dq_model.pkl       sklearn LogisticRegression (replaces existing)
  data/dq_scaler.pkl      sklearn StandardScaler
  metrics/dq_metrics.json accuracy + AUC-ROC
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Project root on sys.path ──────────────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

FEATURE_NAMES = [
    # state
    "ball_x", "ball_y", "dist_to_goal", "angle_to_goal",
    "nearest_defender_dist", "num_defenders_close", "opponent_density",
    "defensive_compactness", "nearest_teammate_dist", "defenders_ahead",
    # candidate
    "target_x", "target_y", "distance",
    # opponent
    "cand_nearest_def_dist", "cand_avg_top2_def_dist",
    "cand_num_defenders_near", "cand_num_defenders_in_lane",
    "cand_min_def_dist_to_lane", "cand_defenders_ahead",
]


# ── Synthetic data generation ─────────────────────────────────────────────────

def _synthetic_dataset(n: int = 20_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    ball_x = rng.uniform(0.0, 1.0, n)
    ball_y = rng.uniform(0.0, 1.0, n)
    dist_to_goal = np.sqrt((ball_x - 1.0) ** 2 + (ball_y - 0.5) ** 2)
    angle_to_goal = np.arctan2(np.abs(0.5 - ball_y), np.abs(1.0 - ball_x))
    nearest_def = rng.uniform(0.02, 0.40, n)
    num_close = rng.integers(0, 5, n).astype(float)
    opp_density = rng.integers(2, 9, n).astype(float)
    compactness = rng.uniform(0.05, 0.30, n)
    nearest_tm = rng.uniform(0.03, 0.35, n)
    def_ahead = rng.integers(0, 6, n).astype(float)

    target_x = rng.uniform(0.0, 1.0, n)
    target_y = rng.uniform(0.0, 1.0, n)
    cand_dist = np.sqrt((ball_x - target_x) ** 2 + (ball_y - target_y) ** 2)

    c_near_def = rng.uniform(0.01, 0.30, n)
    c_avg2 = c_near_def + rng.uniform(0.0, 0.10, n)
    c_num_near = rng.integers(0, 5, n).astype(float)
    c_in_lane = rng.integers(0, 4, n).astype(float)
    c_min_lane = rng.uniform(0.0, 0.20, n)
    c_def_ahead = rng.integers(0, 5, n).astype(float)

    X = np.column_stack([
        ball_x, ball_y, dist_to_goal, angle_to_goal,
        nearest_def, num_close, opp_density, compactness, nearest_tm, def_ahead,
        target_x, target_y, cand_dist,
        c_near_def, c_avg2, c_num_near, c_in_lane, c_min_lane, c_def_ahead,
    ])

    # Success probability: higher when candidate is further from defenders,
    # closer to goal, and few opponents in the lane — roughly physical realism.
    log_odds = (
        -1.5 * dist_to_goal
        + 0.8 * c_near_def
        - 0.4 * c_num_near
        - 0.3 * c_in_lane
        + 0.5 * (1.0 - cand_dist)
        + rng.normal(0, 0.5, n)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    y = (rng.uniform(size=n) < prob).astype(int)
    return X, y


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import pandas as pd
        df = pd.read_csv(path)
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing or "success" not in df.columns:
            print(f"[train_dq] CSV missing columns {missing + ['success']} — using synthetic data")
            return None
        X = df[FEATURE_NAMES].to_numpy(dtype=float)
        y = df["success"].to_numpy(dtype=int)
        return X, y
    except Exception as exc:
        print(f"[train_dq] Could not load CSV: {exc} — using synthetic data")
        return None


def _mlflow_log(run_name: str, params: dict, metrics: dict, artifacts: list[Path]) -> None:
    try:
        import mlflow  # noqa: PLC0415

        uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("dq-model-training")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for p in artifacts:
                if p.exists():
                    mlflow.log_artifact(str(p))
        print("[train_dq] MLflow run logged successfully.")
    except Exception as exc:
        print(f"[train_dq] MLflow logging skipped ({exc}).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Decision Quality model")
    parser.add_argument("--dataset", default="data/final_dataset.csv")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    loaded = _load_csv(dataset_path)
    if loaded is None:
        print(f"[train_dq] Generating {args.n_samples:,} synthetic training samples…")
        X, y = _synthetic_dataset(n=args.n_samples, seed=args.seed)
    else:
        X, y = loaded
        print(f"[train_dq] Loaded {len(X):,} samples from {dataset_path}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=args.seed)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    accuracy = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))

    metrics = {"accuracy": round(accuracy, 4), "auc_roc": round(auc, 4), "n_train": len(X_train), "n_test": len(X_test)}
    params = {"solver": "lbfgs", "C": 1.0, "n_features": X.shape[1], "seed": args.seed}

    model_path = output_dir / "dq_model.pkl"
    scaler_path = output_dir / "dq_scaler.pkl"
    metrics_path = metrics_dir / "dq_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train_dq] accuracy={accuracy:.4f}  auc={auc:.4f}")
    print(f"[train_dq] Saved model → {model_path}")
    print(f"[train_dq] Saved scaler → {scaler_path}")
    print(f"[train_dq] Saved metrics → {metrics_path}")

    _mlflow_log(
        run_name=f"dq-seed{args.seed}",
        params=params,
        metrics=metrics,
        artifacts=[model_path, scaler_path, metrics_path],
    )


if __name__ == "__main__":
    main()
