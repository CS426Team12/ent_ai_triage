import httpx
from app.config import settings
from datetime import date

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

    payload = {
        "patientID": patient_id,
        "transcript": transcript,
        "AISummary": summary,
        "AIUrgency": urgency,
        "AIConfidence": confidence,

        # REQUIRED
        "status": "pending",
        "clinicianSummary": "",
        "overrideSummary": "",
        "overrideUrgency": "",

        # REQUIRED BY DB (createdBy cannot be null)
        "createdBy": "ddc37bcf-50e7-4429-ac7a-425804383b4d",
        "dateCreated": date.today().isoformat()
    }


    headers = {"Authorization": f"Bearer {token}"}

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
