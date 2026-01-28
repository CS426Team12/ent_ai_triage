from fastapi import APIRouter
from pydantic import BaseModel

from app.ollama_client import call_ollama
from app.backend_client import save_triage_to_backend, get_patient_history
from app.ml_client import predict_urgency

router = APIRouter(prefix="/ai")

class TriageRequest(BaseModel):
    transcript: str
    patient_id: str  # required for saving into backend DB


class Flag(BaseModel):
    tag: str  # SYMPTOM, SEVERITY, PROGRESSION, RED_FLAG, MEDICAL_HISTORY, etc.
    keyword: str  # The actual keyword/finding


class TriageResponse(BaseModel):
    summary: str
    urgency: str
    findings: list = []
    flags: list[Flag] = []  # Tagged keywords explaining the decision
    reasoning: str = ""
    ml_confidence: float = 0.0  # Secondary ML validation


@router.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest):
    
    # Fetch patient history for context
    patient_history = await get_patient_history(payload.patient_id)
    print(f"✓ Fetched patient history for {payload.patient_id}")
    
    # Call LLM with transcript + patient history
    # LLM now does BOTH summary generation AND urgency classification
    llm_result = await call_ollama(payload.transcript, patient_history)
    
    summary = llm_result.get("summary", "")
    urgency = llm_result.get("urgency", "routine")
    findings = llm_result.get("findings", [])
    flags_data = llm_result.get("flags", [])
    reasoning = llm_result.get("reasoning", "")
    
    # Convert flags to Flag objects for proper typing
    flags = [Flag(**flag) for flag in flags_data]
    
    # Optional: Get ML model's confidence as secondary validation
    ml_prediction = predict_urgency(payload.transcript)
    ml_confidence = ml_prediction.get("confidence", 0.0)
    
    print(f"📋 LLM Urgency: {urgency}")
    print(f"🚩 Flags detected: {len(flags)}")
    print(f"🤖 ML Confidence: {ml_confidence:.2%}")
    
    # Save triage result to backend
    try:
        await save_triage_to_backend(
            patient_id=payload.patient_id,
            transcript=payload.transcript,
            summary=summary,
            urgency=urgency,
            confidence=ml_confidence,
        )
    except Exception as e:
        print(f"⚠️ Failed to save to backend: {e}")
        # Continue anyway - still return the triage result

    return {
        "summary": summary,
        "urgency": urgency,
        "findings": findings,
        "flags": flags,
        "reasoning": reasoning,
        "ml_confidence": ml_confidence
    }
