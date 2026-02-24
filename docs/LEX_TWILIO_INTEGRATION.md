# Lex + Twilio → AI Triage Integration

This doc is for whoever is wiring Lex/Twilio call data into the AI triage service.

## Overview

- **Your flow:** Twilio number → Lambda webhook → Lex (asks intake Qs, records answers) → your Lambda gets Lex data
- **Our API:** Accepts transcript (or Lex slots), returns urgency + summary + findings
- **Contract:** Your Lambda forwards `{ transcript, patient_id? }` to our triage API

## Endpoints

### 1. `POST /ai/triage` (main)

**Request:**
```json
{
  "transcript": "AI Agent: How long have you had symptoms? Caller: About 3 days. Agent: How severe? Caller: Mild sore throat, maybe 4/10.",
  "patient_id": "unknown"
}
```

- **transcript** (required): Full conversation or symptom description. Lex Q&A, call recording transcript, or free text.
- **patient_id** (optional, default `"unknown"`): Use when you don't have patient verification yet. We skip patient history lookup for `"unknown"`.

**Response:**
```json
{
  "summary": "Patient with mild sore throat, 3 days duration...",
  "urgency": "routine",
  "findings": ["Mild sore throat", "3-day duration"],
  "flags": [{"tag": "SYMPTOM", "keyword": "sore throat"}],
  "reasoning": "Mild symptoms, no red flags...",
  "ml_confidence": 0.92,
  "urgency_confidence": "high"
}
```

### 2. `POST /ai/triage/from-slots` (Lex slots)

Use when Lex sends structured slots instead of a single transcript.

**Request:**
```json
{
  "slots": {
    "symptom": "sore throat",
    "duration": "3 days",
    "severity": "mild"
  },
  "patient_id": "unknown"
}
```

We build a transcript from slots: `"Patient reports: Symptom: sore throat; Duration: 3 days; Severity: mild"`.

### 3. `GET /health`

Returns `{"ok": true}` for liveness checks.

## What to Send From Your Lambda

After Lex finishes the dialog, you have two options:

### Option A: Single transcript string

If you concatenate Lex turns into one string:
```python
# Example: Lex slots or session attributes → transcript string
transcript = "AI: What are your symptoms? Caller: Sore throat. AI: How long? Caller: 3 days."
payload = {"transcript": transcript, "patient_id": "unknown"}
```

### Option B: Structured slots

If you have separate slots (symptom, duration, etc.):
```python
payload = {
    "slots": {"symptom": "...", "duration": "...", "severity": "..."},
    "patient_id": "unknown"
}
# POST to /ai/triage/from-slots
```

## Example Lambda → Triage API Call

```python
import httpx

TRIAGE_API_URL = "https://your-triage-api.com"  # or EC2 URL

def call_triage(transcript: str, patient_id: str = "unknown"):
    r = httpx.post(
        f"{TRIAGE_API_URL}/ai/triage",
        json={"transcript": transcript, "patient_id": patient_id},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()
```

## Patient Verification

- No verification yet: use `patient_id: "unknown"`. We skip patient history and still return triage.
- Later, when you have a real patient ID (from your backend or lookup): pass it and we’ll fetch history for context.

## Environment (for our Lambda / your reference)

- `TRIAGE_API_URL`: Base URL of the triage API (e.g. `https://your-ec2:8100` or `https://api.yourdomain.com`).
- `TRIAGE_API_KEY`: Optional; if set, we send `Authorization: Bearer {key}`.
