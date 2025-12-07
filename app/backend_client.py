import httpx
from app.config import settings

# Global token cache
SERVICE_TOKEN = None

async def get_service_token():
    """
    Automatically logs in to the backend using the AI service account.
    Caches the token so login happens only once.
    """
    global SERVICE_TOKEN

    if SERVICE_TOKEN:
        return SERVICE_TOKEN

    login_payload = {
        "email": settings.BACKEND_USERNAME,
        "password": settings.BACKEND_PASSWORD,
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.BACKEND_BASE_URL}/auth/login",
            json=login_payload
        )
        resp.raise_for_status()
        data = resp.json()

    SERVICE_TOKEN = data["access_token"]
    return SERVICE_TOKEN


async def save_triage_to_backend(
        patient_id: str,
        transcript: str,
        summary: str,
        urgency: str,
        confidence: float
    ):
    """
    Creates a new triage-case record in the backend.
    """
    token = await get_service_token()  # auto-login

    payload = {
        "patientID": patient_id,
        "transcript": transcript,
        "AIConfidence": confidence,
        "AISummary": summary,
        "AIUrgency": urgency
    }

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.BACKEND_BASE_URL}/triage-cases/",
            json=payload,
            headers=headers
        )

        # If token expired → retry automatically once
        if resp.status_code == 401:
            SERVICE_TOKEN = None
            token = await get_service_token()
            headers = {"Authorization": f"Bearer {token}"}

            resp = await client.post(
                f"{settings.BACKEND_BASE_URL}/triage-cases/",
                json=payload,
                headers=headers
            )

        resp.raise_for_status()
