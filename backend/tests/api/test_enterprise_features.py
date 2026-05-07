from fastapi.testclient import TestClient

from app.config.settings import settings
from app.main import app


def test_health_reports_enterprise_feature_flags():
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enterprise_features"]["request_ids"] is True
    assert payload["enterprise_features"]["rate_limiting"]["enabled"] is True
    assert payload["enterprise_features"]["rate_limiting"]["limit"] == settings.RATE_LIMIT_REQUESTS


def test_security_headers_and_request_id_are_present():
    client = TestClient(app)
    response = client.get("/health")

    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Cache-Control"] == "no-store"
    assert "X-Request-ID" in response.headers


def test_rate_limiting_can_be_enforced(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 60)

    first = client.get("/matches")
    second = client.get("/matches")

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded"


def test_ai_analysis_requires_api_key_when_enabled(monkeypatch):
    client = TestClient(app)
    monkeypatch.setattr(settings, "ENTERPRISE_ENFORCE_API_KEY", True)
    monkeypatch.setattr(settings, "ENTERPRISE_API_KEY", "secret-key")
    try:
        denied = client.post("/match/999/ai-analysis")
        allowed = client.post("/match/999/ai-analysis", headers={"X-API-Key": "secret-key"})
    finally:
        monkeypatch.setattr(settings, "ENTERPRISE_ENFORCE_API_KEY", False)
        monkeypatch.setattr(settings, "ENTERPRISE_API_KEY", None)

    assert denied.status_code == 401
    assert allowed.status_code in {404, 500}
