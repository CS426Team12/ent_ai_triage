"""
Unit tests for app.backend_client: _is_known_patient, get_patient_history,
save_triage_to_backend (with mocks).
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


# _is_known_patient is a module-level function
def test_is_known_patient_false_for_unknown():
    from app.backend_client import _is_known_patient
    assert _is_known_patient("unknown") is False
    assert _is_known_patient("null") is False
    assert _is_known_patient("") is False
    assert _is_known_patient("  ") is False
    assert _is_known_patient("None") is False


def test_is_known_patient_true_for_real_id():
    from app.backend_client import _is_known_patient
    assert _is_known_patient("PAT123") is True
    assert _is_known_patient("  PAT123  ") is True


@pytest.mark.asyncio
async def test_get_patient_history_returns_empty_for_unknown():
    from app.backend_client import get_patient_history
    result = await get_patient_history("unknown")
    assert result == {"medicalHistory": [], "allergies": [], "previousVisits": []}


@pytest.mark.asyncio
async def test_get_patient_history_calls_backend_when_known():
    with patch("app.backend_client.get_service_token", new_callable=AsyncMock) as m_token:
        m_token.return_value = "fake-token"
        with patch("app.backend_client.httpx.AsyncClient") as m_client:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "medicalHistory": ["asthma"],
                "allergies": [],
                "previousVisits": [],
            }
            m_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            m_client.return_value.__aexit__.return_value = None

            from app.backend_client import get_patient_history
            result = await get_patient_history("PAT123")
            assert result["medicalHistory"] == ["asthma"]


@pytest.mark.asyncio
async def test_save_triage_to_backend_normalizes_urgency():
    with patch("app.backend_client.get_service_token", new_callable=AsyncMock) as m_token:
        m_token.return_value = "fake-token"
        with patch("app.backend_client.httpx.AsyncClient") as m_client:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            m_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_resp)
            m_client.return_value.__aexit__.return_value = None

            from app.backend_client import save_triage_to_backend
            await save_triage_to_backend(
                patient_id="PAT1",
                transcript="test",
                summary="s",
                urgency="invalid-urgency",
                confidence=0.9,
            )
            call_args = m_client.return_value.__aenter__.return_value.post.call_args
            payload = call_args[1]["json"]
            assert payload["AIUrgency"] == "routine"  # normalized to valid enum
