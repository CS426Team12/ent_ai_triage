"""
AWS Lambda Handler for Lex/Twilio Integration

This Lambda function:
1. Receives data from AWS Lex (or Lex+Twilio webhook)
2. Extracts transcript (from slot or builds from slots) and optional patient_id
3. Calls the ENT Triage API
4. Returns urgency classification

Supports:
- Lex v2 sessionState with slots (patientId, transcript)
- Lex slots as Q&A (symptom, duration, severity, etc.) - builds transcript
- Simple webhook payload: {transcript, patient_id?}
"""

import json
import httpx
import os
from datetime import datetime

# Environment variables
TRIAGE_API_URL = os.environ.get("TRIAGE_API_URL", "http://localhost:8100")
TRIAGE_API_KEY = os.environ.get("TRIAGE_API_KEY", "")
PATIENT_ID_SLOT = "patientId"
TRANSCRIPT_SLOT = "transcript"
UNKNOWN_PATIENT = "unknown"

# Slot keys that may contain conversation/transcript (Lex dialog)
SLOT_KEYS_FOR_TRANSCRIPT = [
    "transcript", "symptoms", "symptom", "chiefComplaint",
    "duration", "severity", "conversation", "summary", "answers",
]


def call_triage_api_sync(patient_id: str, transcript: str) -> dict:
    """
    Call the triage API (sync for Lambda).
    Uses POST /ai/triage; patient_id can be "unknown" when no verification.
    """
    payload = {"patient_id": patient_id, "transcript": transcript}
    headers = {"Content-Type": "application/json"}
    if TRIAGE_API_KEY:
        headers["Authorization"] = f"Bearer {TRIAGE_API_KEY}"
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{TRIAGE_API_URL.rstrip('/')}/ai/triage", json=payload, headers=headers)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        print(f"❌ Triage API error: {e}")
        return {
            "urgency": "routine",
            "summary": "Error processing triage",
            "findings": [],
            "flags": [],
            "reasoning": f"API Error: {str(e)}",
        }


def _get_slot_value(slot: dict) -> str:
    """Extract string from Lex slot value."""
    if not slot:
        return ""
    val = slot.get("value", {})
    if isinstance(val, str):
        return val.strip()
    return str(val.get("interpretedValue", val.get("originalValue", "")) or "").strip()


def _build_transcript_from_slots(slots: dict) -> str:
    """Build transcript from Lex slots (Q&A style) when no single transcript slot."""
    parts = []
    for k, v in slots.items():
        s = _get_slot_value(v) if isinstance(v, dict) else str(v or "").strip()
        if not s:
            continue
        label = k.replace("_", " ").replace("-", " ").title()
        parts.append(f"{label}: {s}")
    return " ".join(parts) if parts else ""


def extract_lex_attributes(event: dict) -> tuple[str, str]:
    """
    Extract patient_id and transcript from Lex/Twilio webhook event.
    Returns (patient_id, transcript). patient_id may be UNKNOWN_PATIENT when no verification.
    Supports: Lex v2 slots, simple {transcript, patient_id} payload, sessionAttributes.
    """
    patient_id = UNKNOWN_PATIENT
    transcript = ""

    # 1. Simple webhook: {transcript, patient_id?}
    if "transcript" in event:
        transcript = event.get("transcript", "") or ""
        patient_id = event.get("patient_id") or event.get("patientId") or UNKNOWN_PATIENT
        return (str(patient_id).strip() or UNKNOWN_PATIENT, str(transcript).strip())

    # 2. Lex v2 sessionState
    try:
        session_state = event.get("sessionState", {})
        intent = session_state.get("intent", {})
        slots = intent.get("slots") or {}

        # Patient ID (optional)
        pid_slot = slots.get(PATIENT_ID_SLOT, {})
        if pid_slot:
            patient_id = _get_slot_value(pid_slot) or UNKNOWN_PATIENT

        # Transcript: prefer transcript slot, else build from slots
        trans_slot = slots.get(TRANSCRIPT_SLOT, {})
        transcript = _get_slot_value(trans_slot)
        if not transcript:
            transcript = session_state.get("sessionAttributes", {}).get("transcript", "")
        if not transcript:
            transcript = _build_transcript_from_slots(slots)

        return (patient_id or UNKNOWN_PATIENT, transcript or "")

    except Exception as e:
        print(f"Error extracting Lex attributes: {e}")
        return UNKNOWN_PATIENT, ""


def format_triage_for_connect(triage_result: dict) -> dict:
    """
    Format triage result for display on Connect agent dashboard.
    """
    
    # Create a simple formatted message
    flags_text = ", ".join([f['keyword'] for f in triage_result.get('flags', [])])
    
    return {
        "urgency": triage_result.get("urgency", "unknown").upper(),
        "summary": triage_result.get("summary", ""),
        "flags": flags_text or "No flags detected",
        "reasoning": triage_result.get("reasoning", ""),
        "confidence": f"{triage_result.get('ml_confidence', 0)*100:.0f}%"
    }


def lambda_handler(event, context):
    """
    Main Lambda handler for Lex/Connect integration.
    
    Triggered by:
    - Lex bot fulfillment (when conversation is complete)
    - Connect Contact Lens (when call recording is available)
    """
    
    print(f"📞 Received event: {json.dumps(event)}")
    
    # Extract patient ID and transcript (patient_id may be "unknown" when no verification yet)
    patient_id, transcript = extract_lex_attributes(event)

    if not transcript or not transcript.strip():
        print("❌ Missing transcript")
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "Missing transcript",
                "message": "Transcript (or Lex slots to build it) is required"
            })
        }

    print(f"✓ Processing triage for patient: {patient_id}")
    print(f"✓ Transcript length: {len(transcript)} characters")

    triage_result = call_triage_api_sync(patient_id, transcript)
    
    # Format for Connect dashboard
    formatted_result = format_triage_for_connect(triage_result)
    
    # Add metadata
    response_payload = {
        "statusCode": 200,
        "timestamp": datetime.utcnow().isoformat(),
        "patient_id": patient_id,
        "triage": formatted_result,
        "raw_response": triage_result
    }
    
    print(f"✓ Triage complete: {formatted_result['urgency']}")
    
    # Return for Connect
    return {
        "statusCode": 200,
        "body": json.dumps(response_payload),
        # For Connect session attributes
        "sessionState": event.get("sessionState", {})
    }


# For local testing
if __name__ == "__main__":
    test_event = {
        "sessionState": {
            "intent": {
                "slots": {
                    "patientId": {
                        "value": {
                            "interpretedValue": "92f082d6-aace-4855-a9d3-40b50a82b18f"
                        }
                    },
                    "transcript": {
                        "value": {
                            "interpretedValue": "AI Agent: Hello, I'm here to help. How long have you had this symptom? Patient: About 2 days. Agent: Is your throat pain mild or severe? Patient: Pretty mild, maybe a 3 out of 10."
                        }
                    }
                }
            }
        }
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))
