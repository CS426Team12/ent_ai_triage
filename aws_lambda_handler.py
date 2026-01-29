"""
AWS Lambda Handler for Lex/Connect Integration

This Lambda function:
1. Receives transcripts from AWS Lex
2. Calls the ENT Triage API
3. Returns urgency classification to Connect agent
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


async def invoke_triage_api(patient_id: str, transcript: str) -> dict:
    """
    Call the triage API with patient transcript.
    
    Args:
        patient_id: Patient ID from Lex
        transcript: Full conversation transcript
        
    Returns:
        dict with triage result (urgency, summary, flags, etc.)
    """
    
    payload = {
        "patient_id": patient_id,
        "transcript": transcript
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if TRIAGE_API_KEY:
        headers["Authorization"] = f"Bearer {TRIAGE_API_KEY}"
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TRIAGE_API_URL}/ai/triage",
                json=payload,
                headers=headers
            )
            
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        print(f"❌ Triage API error: {e}")
        return {
            "urgency": "routine",
            "summary": "Error processing triage",
            "findings": [],
            "flags": [],
            "reasoning": f"API Error: {str(e)}"
        }


def extract_lex_attributes(event: dict) -> tuple:
    """
    Extract patient ID and transcript from Lex event.
    
    Lex v2 format:
    {
        "sessionState": {
            "intent": {
                "slots": {
                    "patientId": {"value": {...}},
                    "transcript": {"value": {...}}
                }
            }
        }
    }
    """
    
    try:
        session_state = event.get("sessionState", {})
        intent = session_state.get("intent", {})
        slots = intent.get("slots", {})
        
        # Extract patient ID
        patient_id_slot = slots.get(PATIENT_ID_SLOT, {})
        patient_id = patient_id_slot.get("value", {}).get("interpretedValue")
        
        # Extract transcript
        transcript_slot = slots.get(TRANSCRIPT_SLOT, {})
        transcript = transcript_slot.get("value", {}).get("interpretedValue", "")
        
        # Fallback: check sessionAttributes for transcript
        if not transcript:
            session_attrs = session_state.get("sessionAttributes", {})
            transcript = session_attrs.get("transcript", "")
        
        return patient_id, transcript
        
    except Exception as e:
        print(f"Error extracting Lex attributes: {e}")
        return None, None


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
    
    # Extract patient ID and transcript
    patient_id, transcript = extract_lex_attributes(event)
    
    if not patient_id or not transcript:
        print("❌ Missing patient ID or transcript")
        return {
            "statusCode": 400,
            "body": json.dumps({
                "error": "Missing patient_id or transcript",
                "message": "Both patient ID and transcript are required"
            })
        }
    
    print(f"✓ Processing triage for patient: {patient_id}")
    print(f"✓ Transcript length: {len(transcript)} characters")
    
    # Call triage API (using synchronous wrapper for Lambda)
    import asyncio
    triage_result = asyncio.run(invoke_triage_api(patient_id, transcript))
    
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
