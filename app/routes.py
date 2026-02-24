from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ollama_client import call_ollama, validate_urgency_classification
from app.backend_client import save_triage_to_backend, get_patient_history
from app.ml_client import predict_urgency

router = APIRouter(prefix="/ai")

# Placeholder when no patient verification (e.g. Lex/Twilio intake without lookup)
UNKNOWN_PATIENT_ID = "unknown"


class TriageRequest(BaseModel):
    transcript: str
    patient_id: str = Field(default=UNKNOWN_PATIENT_ID, description="Optional; use 'unknown' when no verification yet")


class TriageFromSlotsRequest(BaseModel):
    """Lex-style slots: build transcript from Q&A for triage. Use when Lex sends structured slots instead of full transcript."""
    slots: dict = Field(description="e.g. {symptom: 'sore throat', duration: '3 days', severity: 'mild'}")
    patient_id: str = Field(default=UNKNOWN_PATIENT_ID, description="Optional")


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
    urgency_confidence: str = "high"  # high/medium/low confidence in classification


def _build_transcript_from_slots(slots: dict) -> str:
    """Build triage transcript from Lex slots (Q&A style)."""
    parts = []
    for k, v in slots.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        label = k.replace("_", " ").replace("-", " ").title()
        parts.append(f"{label}: {v}")
    return "Patient reports: " + "; ".join(parts) if parts else ""


@router.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest):
    patient_id = payload.patient_id or UNKNOWN_PATIENT_ID

    # Fetch patient history only when we have a real patient ID (skip for Lex/Twilio without verification)
    patient_history = {}
    if patient_id.lower() not in ("unknown", "null", ""):
        patient_history = await get_patient_history(patient_id)
        print(f"✓ Fetched patient history for {patient_id}")
    
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
    
    # Validate and potentially adjust urgency based on flags & medical history
    urgency, urgency_confidence = validate_urgency_classification(
        transcript=payload.transcript,
        llm_urgency=urgency,
        flags=flags_data,
        patient_history=patient_history,
        ml_confidence=ml_confidence
    )
    
    print(f"📋 LLM Urgency: {urgency} (Confidence: {urgency_confidence})")
    print(f"🚩 Flags detected: {len(flags)}")
    print(f"🤖 ML Confidence: {ml_confidence:.2%}")
    
    # Save triage result to backend (skip or use placeholder when patient unknown)
    try:
        await save_triage_to_backend(
            patient_id=patient_id,
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
        "ml_confidence": ml_confidence,
        "urgency_confidence": urgency_confidence
    }


@router.post("/triage/from-slots", response_model=TriageResponse)
async def triage_from_slots(payload: TriageFromSlotsRequest):
    """
    Accept Lex-style slots and build transcript for triage.
    Use when Lex/Twilio sends structured Q&A instead of full transcript.
    """
    transcript = _build_transcript_from_slots(payload.slots)
    if not transcript.strip():
        transcript = "Patient provided no symptom details."
    return await triage(TriageRequest(transcript=transcript, patient_id=payload.patient_id or UNKNOWN_PATIENT_ID))
