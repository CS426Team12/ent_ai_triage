import logging
import random
from datetime import datetime, timezone

from fastapi import APIRouter, Body
from pydantic import BaseModel, Field

from app.ollama_client import (
    apply_groq_output_review_if_enabled,
    call_groq_judge,
    call_ollama,
    validate_urgency_classification,
)
from app.backend_client import save_triage_to_backend, get_patient_history
from app.rf_client import predict_rf_urgency

router = APIRouter(prefix="/ai")
logger = logging.getLogger(__name__)

# Placeholder when no patient verification (e.g. Lex/Twilio intake without lookup)
UNKNOWN_PATIENT_ID = "unknown"
# Demo/dev: when client sends unknown/null, pick a random backend patient UUID for save/history
FALLBACK_PATIENT_IDS: tuple[str, ...] = (
    "b5603780-eb62-46c2-a0ba-d4d796b1cb60",
    "feda50dc-6fd6-4eac-9788-37c600bcd4cb",
    "cc95fc5b-7dfe-48ed-9772-979bfd0b1ecc",
    "a9a58f18-3282-40f4-9b07-a5868067f7c4",
    "acb6d9cb-f75c-4220-a692-f840612ec335",
    "d6b0ab6a-ac7b-47ba-bb63-5a5897f74d8f",
    "ac9b7d46-cd00-431f-9096-4f2d285904f9",
    "27236692-52ca-4454-9311-dc8d3db27a0f",
)


def pick_fallback_patient_id() -> str:
    """Random UUID from FALLBACK_PATIENT_IDS when intake has no verified patient."""
    return random.choice(FALLBACK_PATIENT_IDS)


class TriageRequest(BaseModel):
    transcript: str
    patient_id: str = Field(default=UNKNOWN_PATIENT_ID, description="Optional; use 'unknown' when no verification yet")


class TestPipelineRequest(BaseModel):
    """Optional patient_id for full E2E test (backend requires valid UUID). Default: unknown."""
    patient_id: str = Field(default=UNKNOWN_PATIENT_ID)


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
    received_at: str = ""  # ISO timestamp when call was received (confirms call went through)


def _resolve_consensus_urgency(
    llm_result: dict,
    rf_result: dict,
    judge_result: dict | None = None,
) -> tuple[str, str, dict | None]:
    """Return final urgency, source label, and optional judge metadata."""
    llm_urgency = llm_result.get("urgency", "routine")
    rf_urgency = rf_result.get("urgency", "routine")
    if llm_urgency == rf_urgency:
        return llm_urgency, "consensus", None

    if judge_result is None:
        return llm_urgency, "llm_fallback", None
    return judge_result.get("urgency", "urgent"), "judge", judge_result


def _build_transcript_from_slots(slots: dict) -> str:
    """Build triage transcript from Lex slots (Q&A style)."""
    parts = []
    for k, v in slots.items():
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        label = k.replace("_", " ").replace("-", " ").title()
        parts.append(f"{label}: {v}")
    return "Patient reports: " + "; ".join(parts) if parts else ""


def _transcript_preview(text: str, max_len: int = 200) -> str:
    """Truncate transcript for logging; avoid huge logs."""
    if not text or not text.strip():
        return "(empty)"
    s = text.strip()
    return s[:max_len] + "..." if len(s) > max_len else s


@router.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest):
    received_at = datetime.now(timezone.utc).isoformat()
    patient_id = payload.patient_id or UNKNOWN_PATIENT_ID
    # Use fallback when "unknown" so backend save works (demo/development)
    if patient_id.lower() in ("unknown", "null", ""):
        patient_id = pick_fallback_patient_id()
    preview = _transcript_preview(payload.transcript)
    logger.info(
        f"TRIAGE CALL RECEIVED | patient_id={patient_id} | transcript_len={len(payload.transcript)} | "
        f"transcript_preview=\"{preview}\" | received_at={received_at}"
    )

    # Fetch patient history only when we have a real patient ID (skip for Lex/Twilio without verification)
    patient_history = {}
    if patient_id.lower() not in ("unknown", "null", ""):
        patient_history = await get_patient_history(patient_id)
        logger.info(f"Fetched patient history for {patient_id}")
    
    # Call LLM with transcript + patient history
    # LLM now does BOTH summary generation AND urgency classification
    llm_result = await call_ollama(payload.transcript, patient_history)
    await apply_groq_output_review_if_enabled(payload.transcript, patient_history, llm_result)

    summary = llm_result.get("summary", "")
    urgency = llm_result.get("urgency", "routine")
    findings = llm_result.get("findings", [])
    flags_data = llm_result.get("flags", [])
    reasoning = llm_result.get("reasoning", "")
    
    # Convert flags to Flag objects for proper typing
    flags = [Flag(**flag) for flag in flags_data]
    
    rf_result = predict_rf_urgency(payload.transcript)
    ml_confidence = rf_result.get("confidence", 0.0)

    judge_result = None
    if urgency != rf_result.get("urgency", "routine"):
        judge_result = await call_groq_judge(
            transcript=payload.transcript,
            patient_history=patient_history,
            ollama_result=llm_result,
            rf_result=rf_result,
        )
    urgency, urgency_source, judge_result = _resolve_consensus_urgency(
        llm_result=llm_result,
        rf_result=rf_result,
        judge_result=judge_result,
    )
    
    # Validate and potentially adjust urgency based on flags, summary, and medical history
    urgency, urgency_confidence = validate_urgency_classification(
        transcript=payload.transcript,
        llm_urgency=urgency,
        flags=flags_data,
        patient_history=patient_history,
        ml_confidence=ml_confidence,
        summary=summary,
    )
    
    logger.info(
        "LLM/RF consensus | llm=%s rf=%s final=%s source=%s confidence=%s flags=%s",
        llm_result.get("urgency", "routine"),
        rf_result.get("urgency", "routine"),
        urgency,
        urgency_source,
        urgency_confidence,
        len(flags),
    )
    
    # Save triage result to backend (skip or use placeholder when patient unknown)
    try:
        await save_triage_to_backend(
            patient_id=patient_id,
            transcript=payload.transcript,
            summary=summary,
            urgency=urgency,
            confidence=ml_confidence,
            flags=flags_data,
        )
    except Exception as e:
        logger.warning(f"Failed to save to backend: {e}")
        # Continue anyway - still return the triage result

    logger.info(f"TRIAGE COMPLETE | patient_id={patient_id} | urgency={urgency} | returned_at={datetime.now(timezone.utc).isoformat()}")

    return {
        "summary": summary,
        "urgency": urgency,
        "findings": findings,
        "flags": flags,
        "reasoning": reasoning if judge_result is None else judge_result.get("reasoning", reasoning),
        "ml_confidence": ml_confidence,
        "urgency_confidence": urgency_confidence,
        "received_at": received_at,
    }


# Mock transcript simulating Twilio/Lex call (Step 1)
MOCK_TWILIO_TRANSCRIPT = """Caller: I've had severe ear pain for the past three days and now I have dizziness and some hearing loss.
Agent: Are you experiencing fever?
Caller: Yes, around 101°F."""


@router.post("/test-pipeline")
async def test_pipeline(payload: TestPipelineRequest | None = Body(default=None)):
    """
    Local test simulation: Twilio transcript → AI triage → backend API → database.

    Simulates the full production pipeline without a real Twilio call.
    For backend save to succeed, pass a valid patient UUID in patient_id.
    """
    patient_id = (payload.patient_id if payload else UNKNOWN_PATIENT_ID) or UNKNOWN_PATIENT_ID
    if patient_id.lower() in ("unknown", "null", ""):
        patient_id = pick_fallback_patient_id()
    transcript = MOCK_TWILIO_TRANSCRIPT

    logger.info("Step 1: Simulated Twilio transcript received")

    # Step 2: Run existing triage pipeline (same logic as /triage)
    received_at = datetime.now(timezone.utc).isoformat()

    patient_history = {}
    if patient_id.lower() not in ("unknown", "null", ""):
        patient_history = await get_patient_history(patient_id)

    llm_result = await call_ollama(transcript, patient_history)
    await apply_groq_output_review_if_enabled(transcript, patient_history, llm_result)
    summary = llm_result.get("summary", "")
    urgency = llm_result.get("urgency", "routine")
    findings = llm_result.get("findings", [])
    flags_data = llm_result.get("flags", [])
    reasoning = llm_result.get("reasoning", "")
    flags = [Flag(**f) for f in flags_data]
    rf_result = predict_rf_urgency(transcript)
    ml_confidence = rf_result.get("confidence", 0.0)
    judge_result = None
    if urgency != rf_result.get("urgency", "routine"):
        judge_result = await call_groq_judge(
            transcript=transcript,
            patient_history=patient_history,
            ollama_result=llm_result,
            rf_result=rf_result,
        )
    urgency, urgency_source, judge_result = _resolve_consensus_urgency(
        llm_result=llm_result,
        rf_result=rf_result,
        judge_result=judge_result,
    )
    urgency, urgency_confidence = validate_urgency_classification(
        transcript=transcript,
        llm_urgency=urgency,
        flags=flags_data,
        patient_history=patient_history,
        ml_confidence=ml_confidence,
        summary=summary,
    )

    logger.info("Step 2: AI triage generated")

    triage_result = {
        "summary": summary,
        "urgency_level": urgency,
        "confidence_score": ml_confidence,
        "flagged_keywords": [f.get("keyword", f.get("tag", str(f))) for f in flags_data] if flags_data else [f.keyword for f in flags],
        "transcript": transcript,
        "call_id": "test-call-123",
        "timestamp": received_at,
        "findings": findings,
        "reasoning": reasoning if judge_result is None else judge_result.get("reasoning", reasoning),
        "urgency_confidence": urgency_confidence,
        "urgency_source": urgency_source,
    }

    # Step 3 & 4: Authenticate and send to backend
    try:
        backend_saved, status_code, response_body = await save_triage_to_backend(
            patient_id=patient_id,
            transcript=transcript,
            summary=summary,
            urgency=urgency,
            confidence=ml_confidence,
            flags=flags_data,
        )
    except Exception as e:
        logger.error("Step 4: Backend POST failed: %s", e, exc_info=True)
        return {
            "triage_result": triage_result,
            "backend_saved": False,
            "backend_status": "error",
            "backend_response": str(e),
            "message": "Triage generated but backend save failed. See logs.",
        }

    logger.info("Step 4: Triage result sent to backend API")
    logger.info("Step 5: Backend response: %s", f"{status_code} Created" if status_code == 201 else f"{status_code} {response_body[:200] if response_body else ''}")

    return {
        "triage_result": triage_result,
        "backend_saved": backend_saved,
        "backend_status": f"{status_code} Created" if status_code == 201 else (f"{status_code}" if status_code else "skipped"),
        "backend_response_body": response_body[:500] if response_body else None,
        "message": "Case saved successfully" if backend_saved else "Triage generated; backend save skipped (unknown/invalid patient_id). Use valid patient UUID for full E2E.",
    }


@router.post("/triage/from-slots", response_model=TriageResponse)
async def triage_from_slots(payload: TriageFromSlotsRequest):
    """
    Accept Lex-style slots and build transcript for triage.
    Use when Lex/Twilio sends structured Q&A instead of full transcript.
    """
    logger.info(f"TRIAGE FROM-SLOTS RECEIVED | patient_id={payload.patient_id} | slots={payload.slots}")
    transcript = _build_transcript_from_slots(payload.slots)
    if not transcript.strip():
        transcript = "Patient provided no symptom details."
    logger.info(f"TRIAGE FROM-SLOTS | built transcript: \"{_transcript_preview(transcript)}\"")
    return await triage(TriageRequest(transcript=transcript, patient_id=payload.patient_id or UNKNOWN_PATIENT_ID))
