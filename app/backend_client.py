import httpx

BACKEND_URL = "http://localhost:8000"

async def save_triage_to_backend(patient_id: str, summary: str, urgency: str):
    payload = {
        "patient_id": patient_id,
        "summary": summary,
        "urgency": urgency
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{BACKEND_URL}/triage", json=payload)
        resp.raise_for_status()

    return resp.json()
