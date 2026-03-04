"""
Pytest configuration and shared fixtures.
Set required env vars before app imports so Settings() does not fail.
"""
import os
import sys

# Set minimal env for Settings (required by config.py) before any app imports
os.environ.setdefault("BACKEND_BASE_URL", "http://test-backend:8000")
os.environ.setdefault("BACKEND_USERNAME", "test@test.com")
os.environ.setdefault("BACKEND_PASSWORD", "testpass")
os.environ.setdefault("OLLAMA_BASE_URL", "http://test-ollama:11434")

# Ensure app package is on path when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_triage_payload():
    return {
        "transcript": "Patient has mild sore throat for 3 days. No fever.",
        "patient_id": "unknown",
    }


@pytest.fixture
def sample_slots_payload():
    return {
        "slots": {
            "symptom": "sore throat",
            "duration": "3 days",
            "severity": "mild",
        },
        "patient_id": "unknown",
    }
