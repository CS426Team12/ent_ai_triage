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


def _is_known_patient(patient_id: str) -> bool:
    """True if we have a real patient ID (not placeholder from Lex/Twilio without verification)."""
    if not patient_id or not str(patient_id).strip():
        return False
    v = str(patient_id).lower()
    return v not in ("unknown", "null", "none", "")


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
        print(f"⚠️ Failed to fetch patient history: {e}")
        return {
            "medicalHistory": [],
            "allergies": [],
            "previousVisits": []
        }



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
