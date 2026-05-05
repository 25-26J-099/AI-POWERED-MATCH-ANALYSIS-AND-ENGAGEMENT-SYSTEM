from __future__ import annotations

from fastapi import Header, HTTPException

from app.config.settings import settings


async def require_enterprise_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not settings.ENTERPRISE_ENFORCE_API_KEY:
        return
    expected = settings.ENTERPRISE_API_KEY
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing enterprise API key")
