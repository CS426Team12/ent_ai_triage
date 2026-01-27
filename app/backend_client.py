import httpx
from app.config import settings

SERVICE_TOKEN = None

async def get_service_token():
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


async def save_triage_to_backend(patient_id, transcript, summary, urgency, confidence):
    token = await get_service_token()

    # Enforce enum values
    valid_urgencies = {"routine", "semi-urgent", "urgent"}
    if urgency not in valid_urgencies:
        urgency = "routine"

    payload = {
        "patientID": patient_id,
        "transcript": transcript,
        "AISummary": summary,
        "AIUrgency": urgency,
        "AIConfidence": confidence

    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.BACKEND_BASE_URL}/triage-cases/",
            json=payload,
            headers=headers,
        )

    if resp.status_code >= 400:
        print("❌ REAL BACKEND ERROR FROM /triage-cases/")
        print("Status:", resp.status_code)
        print("Body:", resp.text)
        print("Sent payload:", payload)

    resp.raise_for_status()
