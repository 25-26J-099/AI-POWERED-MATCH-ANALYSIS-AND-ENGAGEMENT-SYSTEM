"""Train VAEP scoring and conceding probability models.

VAEP (Valuing Actions by Estimating Probabilities) requires two LightGBM
classifiers:

  1. P_score:   probability that a team scores within 10 actions
  2. P_concede: probability that a team concedes within 10 actions

The VAEP value per action = (P_score_after - P_score_before)
                           - (P_concede_after - P_concede_before)

Features (12) match ``app/analytics/vaep.py``:
  start_x, start_y, end_x, end_y, action_length,
  distance_to_goal, end_distance_to_goal,
  xt_start, xt_end, delta_xt, success, action_type

Outputs
-------
  model_cache/vaep_score_model_2.pkl    LightGBM scoring classifier
  model_cache/vaep_concede_model_2.pkl  LightGBM conceding classifier
  metrics/vaep_metrics.json             AUC-ROC for both models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

FEATURE_NAMES = [
    "start_x", "start_y", "end_x", "end_y", "action_length",
    "distance_to_goal", "end_distance_to_goal",
    "xt_start", "xt_end", "delta_xt", "success", "action_type",
]

# Action type encoding (matches vaep.py)
ACTION_TYPES = {"pass": 0, "carry": 1, "shot": 2, "dribble": 3,
                "interception": 4, "clearance": 5}


def _xt_value(x: float, y: float) -> float:
    """Simple xT approximation: value increases toward opponent goal."""
    return max(0.0, (x - 0.5) * 0.2 + 0.01)


def _synthetic_dataset(n: int = 40_000, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    # Action locations
    sx = rng.uniform(0.0, 1.0, n)
    sy = rng.uniform(0.0, 1.0, n)
    ex = np.clip(sx + rng.normal(0.0, 0.15, n), 0.0, 1.0)
    ey = np.clip(sy + rng.normal(0.0, 0.10, n), 0.0, 1.0)

    length = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
    dist_goal = np.sqrt((1.0 - sx) ** 2 + (0.5 - sy) ** 2)
    end_dist = np.sqrt((1.0 - ex) ** 2 + (0.5 - ey) ** 2)

    # Simple xT lookup
    xt_s = np.array([_xt_value(x, y) for x, y in zip(sx, sy)])
    xt_e = np.array([_xt_value(x, y) for x, y in zip(ex, ey)])
    d_xt = xt_e - xt_s

    success = rng.choice([0, 1], size=n, p=[0.25, 0.75]).astype(float)
    a_type = rng.choice(list(ACTION_TYPES.values()), size=n).astype(float)

    X = np.column_stack([
        sx, sy, ex, ey, length,
        dist_goal, end_dist,
        xt_s, xt_e, d_xt, success, a_type,
    ])

    # Scoring: more likely when action ends closer to goal
    log_score = (
        -1.0 * end_dist
        + 1.5 * d_xt
        + 0.5 * (a_type == ACTION_TYPES["shot"])
        + 0.3 * success
        + rng.normal(0, 0.5, n)
    )
    p_score = 1.0 / (1.0 + np.exp(-log_score))
    y_score = (rng.uniform(size=n) < p_score).astype(int)

    # Conceding: more likely after losing ball far from own goal
    log_concede = (
        1.0 * dist_goal
        - 1.5 * d_xt
        - 0.5 * success
        + rng.normal(0, 0.5, n)
    )
    p_concede = 1.0 / (1.0 + np.exp(-log_concede))
    y_concede = (rng.uniform(size=n) < p_concede).astype(int)

    return X, y_score, y_concede


def _load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        import pandas as pd
        df = pd.read_csv(path)
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        if missing or "label_score" not in df.columns or "label_concede" not in df.columns:
            print("[train_vaep] CSV missing required columns — using synthetic data")
            return None
        X = df[FEATURE_NAMES].to_numpy(dtype=float)
        return X, df["label_score"].to_numpy(int), df["label_concede"].to_numpy(int)
    except Exception as exc:
        print(f"[train_vaep] Could not load CSV: {exc} — using synthetic data")
        return None


def _train_lgbm(X_train, y_train, seed: int):
    try:
        import lightgbm as lgb  # noqa: PLC0415
        model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=seed, verbose=-1,
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        print("[train_vaep] lightgbm not available, falling back to GradientBoostingClassifier")
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=seed)
        model.fit(X_train, y_train)
        return model


def _mlflow_log(params: dict, metrics: dict, artifacts: list[Path]) -> None:
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5000"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("vaep-model-training")
        with mlflow.start_run(run_name="vaep-lgbm"):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for p in artifacts:
                if p.exists():
                    mlflow.log_artifact(str(p))
        print("[train_vaep] MLflow run logged successfully.")
    except Exception as exc:
        print(f"[train_vaep] MLflow logging skipped ({exc}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train VAEP scoring and conceding models")
    parser.add_argument("--dataset", default="data/vaep_dataset.csv")
    parser.add_argument("--output-dir", default="model_cache")
    parser.add_argument("--metrics-dir", default="metrics")
    parser.add_argument("--n-samples", type=int, default=40_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    metrics_dir = Path(args.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(args.dataset)
    loaded = _load_csv(dataset_path)
    if loaded is None:
        print(f"[train_vaep] Generating {args.n_samples:,} synthetic action samples…")
        X, y_score, y_concede = _synthetic_dataset(n=args.n_samples, seed=args.seed)
    else:
        X, y_score, y_concede = loaded
        print(f"[train_vaep] Loaded {len(X):,} samples from {dataset_path}")

    X_tr, X_te, ys_tr, ys_te, yc_tr, yc_te = train_test_split(
        X, y_score, y_concede, test_size=0.2, random_state=args.seed
    )

    print("[train_vaep] Training scoring model…")
    score_model = _train_lgbm(X_tr, ys_tr, args.seed)
    print("[train_vaep] Training conceding model…")
    concede_model = _train_lgbm(X_tr, yc_tr, args.seed)

    auc_score = float(roc_auc_score(ys_te, score_model.predict_proba(X_te)[:, 1]))
    auc_concede = float(roc_auc_score(yc_te, concede_model.predict_proba(X_te)[:, 1]))

    metrics = {
        "auc_score_model": round(auc_score, 4),
        "auc_concede_model": round(auc_concede, 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }
    params = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "seed": args.seed}

    score_path = output_dir / "vaep_score_model_2.pkl"
    concede_path = output_dir / "vaep_concede_model_2.pkl"
    metrics_path = metrics_dir / "vaep_metrics.json"

    joblib.dump(score_model, score_path)
    joblib.dump(concede_model, concede_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(f"[train_vaep] auc_score={auc_score:.4f}  auc_concede={auc_concede:.4f}")
    print(f"[train_vaep] Saved → {score_path}, {concede_path}")
    print(f"[train_vaep] Saved metrics → {metrics_path}")

    _mlflow_log(
        params=params,
        metrics=metrics,
        artifacts=[score_path, concede_path, metrics_path],
    )


if __name__ == "__main__":
    main()
