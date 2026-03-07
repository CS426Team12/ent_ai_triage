import logging
import uuid
import httpx
from app.config import settings

logger = logging.getLogger(__name__)
SERVICE_TOKEN = None


async def get_service_token() -> str:
    """Authenticate with backend via JWT login. Returns access token."""
    global SERVICE_TOKEN

    if SERVICE_TOKEN:
        return SERVICE_TOKEN

    login_payload = {
        "email": settings.BACKEND_USERNAME,
        "password": settings.BACKEND_PASSWORD,
    }
    logger.info("Authenticating with backend at %s/auth/login", settings.BACKEND_BASE_URL)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{settings.BACKEND_BASE_URL}/auth/login",
                json=login_payload,
            )
            resp.raise_for_status()
            data = resp.json()
        SERVICE_TOKEN = data["access_token"]
        logger.info("Backend auth successful")
        return SERVICE_TOKEN
    except Exception as e:
        logger.error("Backend auth failed: %s", e, exc_info=True)
        raise


def _is_known_patient(patient_id: str) -> bool:
    """True if we have a real patient ID (not placeholder from Lex/Twilio without verification)."""
    if not patient_id or not str(patient_id).strip():
        return False
    v = str(patient_id).lower()
    return v not in ("unknown", "null", "none", "")


def _is_valid_uuid(value: str) -> bool:
    """True if value is a valid UUID string."""
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError):
        return False


async def get_patient_history(patient_id: str) -> dict:
    """Fetch patient medical history from backend. Returns empty dict when patient_id is unknown/placeholder."""
    if not _is_known_patient(patient_id):
        return {"medicalHistory": [], "allergies": [], "previousVisits": []}
    try:
        token = await get_service_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{settings.BACKEND_BASE_URL}/patients/{patient_id}",
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.warning("Failed to fetch patient history: %s", e)
        return {
            "medicalHistory": [],
            "allergies": [],
            "previousVisits": []
        }



def _map_ai_to_backend_payload(
    patient_id: str,
    transcript: str,
    summary: str,
    urgency: str,
    confidence: float,
) -> dict:
    """
    Map AI triage output to backend TriageCaseCreate schema.

    Backend expects:
    - patientID: UUID
    - transcript: str
    - AISummary: str
    - AIUrgency: str (routine | semi-urgent | urgent)
    - AIConfidence: float
    """
    valid_urgencies = {"routine", "semi-urgent", "urgent"}
    ai_urgency = (urgency or "").lower().strip()
    if ai_urgency not in valid_urgencies:
        ai_urgency = "routine"

    return {
        "patientID": str(patient_id),
        "transcript": transcript or "",
        "AISummary": summary or "",
        "AIUrgency": ai_urgency,
        "AIConfidence": confidence,
    }


async def save_triage_to_backend(
    patient_id: str,
    transcript: str,
    summary: str,
    urgency: str,
    confidence: float,
) -> tuple[bool, int | None, str]:
    """
    Send AI triage result to backend API.

    Returns (success, status_code, response_body).
    - success=True, status_code=201, body="..." when saved
    - success=False, status_code=None, body="skipped" when patient unknown/invalid
    - Raises on HTTP/network error
    """
    if not _is_known_patient(patient_id):
        logger.info(
            "Skipping backend save: patient_id=%s (unknown/placeholder); backend requires valid patient UUID",
            patient_id,
        )
        return False, None, "skipped (unknown patient)"

    if not _is_valid_uuid(patient_id):
        logger.warning(
            "Skipping backend save: patient_id=%s is not a valid UUID",
            patient_id,
        )
        return False, None, "skipped (invalid UUID)"

    try:
        token = await get_service_token()
        payload = _map_ai_to_backend_payload(
            patient_id=patient_id,
            transcript=transcript,
            summary=summary,
            urgency=urgency,
            confidence=confidence,
        )

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{settings.BACKEND_BASE_URL}/triage-cases/",
                json=payload,
                headers=headers,
            )

        if resp.status_code >= 400:
            logger.error(
                "Backend POST /triage-cases/ failed: status=%d, body=%s, payload=%s",
                resp.status_code,
                resp.text,
                payload,
            )
            resp.raise_for_status()

        logger.info(
            "Triage result saved to backend: patient_id=%s, urgency=%s",
            patient_id,
            payload.get("AIUrgency"),
        )
        return True, resp.status_code, resp.text

    except httpx.HTTPStatusError as e:
        logger.error(
            "Backend request failed: %s %s - %s",
            e.request.method,
            e.request.url,
            e.response.text,
            exc_info=True,
        )
        raise
    except Exception as e:
        logger.error("Failed to save triage to backend: %s", e, exc_info=True)
        raise
