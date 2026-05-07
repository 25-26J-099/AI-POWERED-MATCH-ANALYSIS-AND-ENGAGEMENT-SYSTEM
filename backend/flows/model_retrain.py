"""
Model Retraining Pipeline — Prefect flow.

Scheduled to run weekly (or triggered manually via Prefect UI / CLI).
Retrains xG, VAEP, DQ, and style models, logs results to MLflow,
and pushes updated artifacts to DVC remote (GCS).

Schedule: every Sunday at 02:00 UTC.

Deploy to Prefect Cloud:
    prefect deploy backend/flows/model_retrain.py:retrain_all_models \
        --name weekly-retrain \
        --cron "0 2 * * 0"
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from prefect import flow, task, get_run_logger

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="dvc-pull", retries=2, retry_delay_seconds=30)
def dvc_pull_task() -> None:
    """Pull latest data and artifacts from DVC remote (GCS)."""
    run_logger = get_run_logger()
    run_logger.info("Pulling DVC artifacts from GCS...")
    result = subprocess.run(
        ["dvc", "pull", "--force"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        run_logger.warning("dvc pull stderr: %s", result.stderr)
    else:
        run_logger.info("dvc pull stdout: %s", result.stdout)


@task(name="train-xg", timeout_seconds=3600)
def train_xg_task() -> dict:
    """Retrain the xG model and log to MLflow."""
    import mlflow

    run_logger = get_run_logger()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("xg-model")

    run_logger.info("Training xG model...")
    result = subprocess.run(
        ["python", "scripts/train_xg.py"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    run_logger.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"xG training failed:\n{result.stderr}")

    # Read metrics produced by the training script
    metrics_path = BACKEND_DIR / "metrics" / "xg_metrics.json"
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
        run_logger.info("xG metrics: %s", metrics)
        return metrics
    return {}


@task(name="train-vaep", timeout_seconds=3600)
def train_vaep_task() -> dict:
    """Retrain VAEP scoring and conceding models and log to MLflow."""
    run_logger = get_run_logger()
    run_logger.info("Training VAEP models...")
    result = subprocess.run(
        ["python", "scripts/train_vaep.py"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    run_logger.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"VAEP training failed:\n{result.stderr}")

    metrics_path = BACKEND_DIR / "metrics" / "vaep_metrics.json"
    if metrics_path.exists():
        import json
        return json.loads(metrics_path.read_text())
    return {}


@task(name="train-dq", timeout_seconds=3600)
def train_dq_task() -> dict:
    """Retrain the Decision Quality model and log to MLflow."""
    run_logger = get_run_logger()
    run_logger.info("Training DQ model...")
    result = subprocess.run(
        ["python", "scripts/train_dq.py"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    run_logger.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"DQ training failed:\n{result.stderr}")

    metrics_path = BACKEND_DIR / "metrics" / "dq_metrics.json"
    if metrics_path.exists():
        import json
        return json.loads(metrics_path.read_text())
    return {}


@task(name="train-style", timeout_seconds=3600)
def train_style_task() -> dict:
    """Retrain style scaler, autoencoder, and KMeans cluster model."""
    run_logger = get_run_logger()
    run_logger.info("Training style embedding models...")
    result = subprocess.run(
        ["python", "scripts/train_style.py"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    run_logger.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Style training failed:\n{result.stderr}")

    metrics_path = BACKEND_DIR / "metrics" / "style_metrics.json"
    if metrics_path.exists():
        import json
        return json.loads(metrics_path.read_text())
    return {}


@task(name="dvc-push", retries=2, retry_delay_seconds=30)
def dvc_push_task() -> None:
    """Push updated model artifacts back to DVC remote (GCS)."""
    run_logger = get_run_logger()
    run_logger.info("Pushing updated artifacts to GCS via DVC...")
    result = subprocess.run(
        ["dvc", "push"],
        cwd=BACKEND_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc push failed:\n{result.stderr}")
    run_logger.info("dvc push complete: %s", result.stdout)


@task(name="invalidate-model-cache")
def invalidate_cache_task() -> None:
    """Clear in-process model caches so the next request loads fresh weights."""
    try:
        import app.services.model_loader as ml
        for attr in ("_xg_model", "_vaep_scoring_model", "_vaep_conceding_model",
                     "_style_scaler", "_style_autoencoder", "_style_kmeans"):
            setattr(ml, attr, ml._UNSET)
        get_run_logger().info("In-process model caches cleared")
    except Exception as exc:
        get_run_logger().warning("Could not clear model caches: %s", exc)


# ---------------------------------------------------------------------------
# Main retraining flow
# ---------------------------------------------------------------------------

@flow(
    name="retrain-all-models",
    description="Weekly retraining of xG, VAEP, DQ, and style models with DVC + MLflow tracking",
    log_prints=True,
)
def retrain_all_models() -> dict:
    """Pull data → retrain all models in parallel → push artifacts → invalidate cache."""
    run_logger = get_run_logger()
    run_logger.info("=== Weekly model retraining started ===")

    # Step 1: pull latest data/artifacts from GCS
    dvc_pull_task()

    # Step 2: retrain all models in parallel
    xg_future    = train_xg_task.submit()
    vaep_future  = train_vaep_task.submit()
    dq_future    = train_dq_task.submit()
    style_future = train_style_task.submit()

    xg_metrics    = xg_future.result()
    vaep_metrics  = vaep_future.result()
    dq_metrics    = dq_future.result()
    style_metrics = style_future.result()

    # Step 3: push updated artifacts to GCS
    dvc_push_task()

    # Step 4: invalidate in-process model caches (backend picks up new weights)
    invalidate_cache_task()

    run_logger.info("=== Retraining complete ===")
    return {
        "xg": xg_metrics,
        "vaep": vaep_metrics,
        "dq": dq_metrics,
        "style": style_metrics,
    }


if __name__ == "__main__":
    retrain_all_models()
