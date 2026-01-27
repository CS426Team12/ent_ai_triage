from fastapi import APIRouter
from pydantic import BaseModel

from app.ollama_client import call_ollama
from app.backend_client import save_triage_to_backend
from app.ml_client import predict_urgency

router = APIRouter(prefix="/ai")

class TriageRequest(BaseModel):
    transcript: str
    patient_id: str # required for saving into backend DB


class TriageResponse(BaseModel):
    summary: str
    urgency: str
    ml_confidence: float = 0.0


@router.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest):

    raw = await call_ollama(payload.transcript)

    summary = raw.strip()
    
    # Use ML model to predict urgency
    ml_prediction = predict_urgency(payload.transcript)
    urgency = ml_prediction["urgency"]
    confidence = ml_prediction["confidence"]

    try:
        await save_triage_to_backend(
            patient_id=payload.patient_id,
            transcript=payload.transcript,
            summary=summary,
            urgency=urgency,
            confidence=confidence,
        )
    except Exception as e:
        print(f"⚠️ Failed to save to backend: {e}")
        # Continue anyway - still return the triage result

    return {
        "summary": summary,
        "urgency": urgency,
        "ml_confidence": confidence
    }