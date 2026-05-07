"""GCS-backed file storage utility.

When GCS_BUCKET is set in the environment, uploaded videos and commentary
output are mirrored to Cloud Storage after being written locally.
When GCS_BUCKET is not set (local dev), files stay on disk only.

Usage:
    from app.utils.storage import upload_to_gcs, gcs_url

    # After saving to local path:
    gcs_key = f"uploads/{match_id}/{filename}"
    public_url = upload_to_gcs(local_path, gcs_key)   # None if no GCS bucket
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GCS_CLIENT = None  # lazy singleton


def _get_gcs_client():
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        try:
            from google.cloud import storage as gcs
            _GCS_CLIENT = gcs.Client()
        except Exception as exc:
            logger.warning("GCS client unavailable: %s", exc)
            _GCS_CLIENT = False  # mark as unavailable so we don't retry
    return _GCS_CLIENT if _GCS_CLIENT else None


def upload_to_gcs(local_path: str | Path, gcs_key: str) -> Optional[str]:
    """Upload a local file to GCS and return the gs:// URL.

    Returns None if GCS_BUCKET is not configured or upload fails.
    The local file is NOT deleted — it remains as a local cache.
    """
    bucket_name = os.environ.get("GCS_BUCKET", "")
    if not bucket_name:
        return None  # local-dev mode, skip GCS

    client = _get_gcs_client()
    if client is None:
        return None

    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path))
        gcs_url = f"gs://{bucket_name}/{gcs_key}"
        logger.info("Uploaded %s → %s", local_path, gcs_url)
        return gcs_url
    except Exception as exc:
        logger.warning("GCS upload failed for %s: %s", gcs_key, exc)
        return None


def download_from_gcs(gcs_key: str, local_path: str | Path) -> bool:
    """Download a file from GCS to a local path.

    Returns True on success, False on failure or if GCS is not configured.
    """
    bucket_name = os.environ.get("GCS_BUCKET", "")
    if not bucket_name:
        return False

    client = _get_gcs_client()
    if client is None:
        return False

    try:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(gcs_key)
        blob.download_to_filename(str(local_path))
        logger.info("Downloaded gs://%s/%s → %s", bucket_name, gcs_key, local_path)
        return True
    except Exception as exc:
        logger.warning("GCS download failed for %s: %s", gcs_key, exc)
        return False


def gcs_url(gcs_key: str) -> Optional[str]:
    """Return a gs:// URL for the given key, or None if no bucket is configured."""
    bucket_name = os.environ.get("GCS_BUCKET", "")
    return f"gs://{bucket_name}/{gcs_key}" if bucket_name else None
