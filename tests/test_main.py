"""
Unit tests for app.main: FastAPI app, health endpoint, middleware.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app, health


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_and_status(self, client: TestClient):
        response = client.get("/health")
        data = response.json()
        assert data["ok"] is True
        assert data["status"] == "running"

    def test_health_direct_call(self):
        result = health()
        assert result == {"ok": True, "status": "running"}


class TestAppSetup:
    """Tests for FastAPI app configuration."""

    def test_app_title(self):
        assert app.title == "AI Triage Service"

    def test_app_has_health_route(self):
        routes = [r.path for r in app.routes]
        assert "/health" in routes

    def test_app_includes_ai_routes(self):
        # Router prefix is /ai, so we get /ai/triage etc.
        routes = [r.path for r in app.routes]
        assert any("/ai" in p for p in routes)
