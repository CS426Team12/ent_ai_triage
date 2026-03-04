"""
Unit tests for app.routes: triage endpoints, request/response models, helpers.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from app.routes import (
    _build_transcript_from_slots,
    _transcript_preview,
    TriageRequest,
    TriageFromSlotsRequest,
    Flag,
    TriageResponse,
    UNKNOWN_PATIENT_ID,
)


class TestBuildTranscriptFromSlots:
    """Tests for _build_transcript_from_slots."""

    def test_builds_transcript_from_slots(self):
        slots = {"symptom": "sore throat", "duration": "3 days", "severity": "mild"}
        result = _build_transcript_from_slots(slots)
        assert "Patient reports:" in result
        assert "sore throat" in result
        assert "3 days" in result
        assert "mild" in result

    def test_skips_none_and_empty(self):
        slots = {"symptom": "cough", "duration": None, "severity": ""}
        result = _build_transcript_from_slots(slots)
        assert "cough" in result
        assert "None" not in result or "duration" not in result

    def test_empty_slots_returns_empty_string(self):
        result = _build_transcript_from_slots({})
        assert result == ""

    def test_formats_keys_as_title(self):
        slots = {"ear_pain": "yes"}
        result = _build_transcript_from_slots(slots)
        assert "Ear Pain" in result or "ear pain" in result


class TestTranscriptPreview:
    """Tests for _transcript_preview."""

    def test_short_text_unchanged(self):
        text = "short"
        assert _transcript_preview(text) == "short"

    def test_long_text_truncated_with_ellipsis(self):
        text = "a" * 250
        result = _transcript_preview(text, max_len=200)
        assert len(result) == 203
        assert result.endswith("...")

    def test_empty_returns_empty_marker(self):
        assert _transcript_preview("") == "(empty)"
        assert _transcript_preview("   ") == "(empty)"

    def test_custom_max_len(self):
        text = "hello world"
        assert _transcript_preview(text, max_len=5) == "hello..."


class TestRequestResponseModels:
    """Tests for Pydantic models."""

    def test_triage_request_default_patient_id(self):
        req = TriageRequest(transcript="test")
        assert req.patient_id == UNKNOWN_PATIENT_ID

    def test_triage_request_accepts_patient_id(self):
        req = TriageRequest(transcript="test", patient_id="PAT123")
        assert req.patient_id == "PAT123"

    def test_flag_model(self):
        f = Flag(tag="SYMPTOM", keyword="sore throat")
        assert f.tag == "SYMPTOM"
        assert f.keyword == "sore throat"

    def test_triage_response_has_required_fields(self):
        r = TriageResponse(
            summary="s", urgency="routine", received_at="2026-01-01T00:00:00Z"
        )
        assert r.summary == "s"
        assert r.urgency == "routine"
        assert r.received_at == "2026-01-01T00:00:00Z"
        assert r.findings == []
        assert r.flags == []
        assert r.ml_confidence == 0.0


class TestTriageEndpoint:
    """Tests for POST /ai/triage (with mocks)."""

    @pytest.fixture(autouse=True)
    def _mock_dependencies(self):
        with (
            patch("app.routes.call_ollama", new_callable=AsyncMock) as m_ollama,
            patch("app.routes.get_patient_history", new_callable=AsyncMock) as m_history,
            patch("app.routes.save_triage_to_backend", new_callable=AsyncMock) as m_save,
            patch("app.routes.predict_urgency") as m_ml,
            patch("app.routes.validate_urgency_classification") as m_validate,
        ):
            m_ollama.return_value = {
                "summary": "Test summary",
                "urgency": "routine",
                "findings": ["Mild sore throat"],
                "flags": [{"tag": "SYMPTOM", "keyword": "sore throat"}],
                "reasoning": "Mild case.",
            }
            m_history.return_value = {}
            m_ml.return_value = {"urgency": "routine", "confidence": 0.9}
            m_validate.return_value = ("routine", "high")
            yield {
                "ollama": m_ollama,
                "history": m_history,
                "save": m_save,
                "ml": m_ml,
                "validate": m_validate,
            }

    def test_triage_returns_200(self, client: TestClient, sample_triage_payload):
        response = client.post("/ai/triage", json=sample_triage_payload)
        assert response.status_code == 200

    def test_triage_returns_expected_shape(self, client: TestClient, sample_triage_payload):
        response = client.post("/ai/triage", json=sample_triage_payload)
        data = response.json()
        assert "summary" in data
        assert "urgency" in data
        assert "findings" in data
        assert "flags" in data
        assert "reasoning" in data
        assert "received_at" in data
        assert "ml_confidence" in data
        assert "urgency_confidence" in data

    def test_triage_unknown_patient_skips_history(
        self, client: TestClient, _mock_dependencies
    ):
        client.post("/ai/triage", json={"transcript": "sore throat", "patient_id": "unknown"})
        _mock_dependencies["history"].assert_not_called()

    def test_triage_valid_patient_calls_history(
        self, client: TestClient, _mock_dependencies
    ):
        client.post(
            "/ai/triage",
            json={"transcript": "sore throat", "patient_id": "PAT123"},
        )
        _mock_dependencies["history"].assert_called_once_with("PAT123")


class TestTriageFromSlotsEndpoint:
    """Tests for POST /ai/triage/from-slots (with mocks)."""

    @pytest.fixture(autouse=True)
    def _mock_dependencies(self):
        with (
            patch("app.routes.call_ollama", new_callable=AsyncMock) as m_ollama,
            patch("app.routes.get_patient_history", new_callable=AsyncMock) as m_history,
            patch("app.routes.save_triage_to_backend", new_callable=AsyncMock) as m_save,
            patch("app.routes.predict_urgency") as m_ml,
            patch("app.routes.validate_urgency_classification") as m_validate,
        ):
            m_ollama.return_value = {
                "summary": "From slots.",
                "urgency": "routine",
                "findings": [],
                "flags": [],
                "reasoning": "Routine.",
            }
            m_ml.return_value = {"urgency": "routine", "confidence": 0.85}
            m_validate.return_value = ("routine", "high")
            yield

    def test_from_slots_returns_200(self, client: TestClient, sample_slots_payload):
        response = client.post("/ai/triage/from-slots", json=sample_slots_payload)
        assert response.status_code == 200

    def test_from_slots_builds_transcript(self, client: TestClient, sample_slots_payload):
        response = client.post("/ai/triage/from-slots", json=sample_slots_payload)
        data = response.json()
        assert data["summary"] == "From slots."
        assert "received_at" in data
