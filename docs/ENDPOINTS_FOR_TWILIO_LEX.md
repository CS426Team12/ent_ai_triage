# Endpoints for Twilio + Amazon Lex Integration

**Base URL:** `http://3.146.34.91:8100`

Use these endpoints when wiring your Twilio/Lex Lambda to the AI triage service.

---

## 1. POST `/ai/triage`

**Use when:** You have a full transcript string (e.g. from Twilio/Lex Q&A).

**Request:**
```json
{
  "transcript": "chief_complaint:I'm coughing. symptom_duration:1 month. symptom_severity:moderate. symptom_progression:better. aggravating_factors:Drinking alcohol. relieving_factors:Physical activity. associated_symptoms:No. red_flags:No. risk_factors:No",
  "patient_id": "unknown"
}
```

| Field        | Required | Description |
|--------------|----------|-------------|
| `transcript` | Yes      | Full transcript or Lex slot string. Twilio format: `slot:value. slot:value.` |
| `patient_id` | No       | Default `"unknown"`. Use real ID when you have patient verification. |

**Response:**
```json
{
  "summary": "Patient with chronic cough of 1 month, moderate severity, now improving...",
  "urgency": "routine",
  "findings": ["Chronic cough", "1-month duration", "Moderate severity"],
  "flags": [{"tag": "SYMPTOM", "keyword": "cough"}, {"tag": "SEVERITY", "keyword": "moderate"}],
  "reasoning": "Chronic cough improving, no red flags. Routine evaluation appropriate.",
  "ml_confidence": 0.92,
  "urgency_confidence": "high",
  "received_at": "2026-02-28T01:30:00.123456+00:00"
}
```

---

## 2. POST `/ai/triage/from-slots`

**Use when:** Lex sends structured slots as a dict.

**Request:**
```json
{
  "slots": {
    "chief_complaint": "sore throat",
    "symptom_duration": "3 days",
    "symptom_severity": "mild",
    "symptom_progression": "improving",
    "aggravating_factors": "swallowing",
    "relieving_factors": "warm tea",
    "associated_symptoms": "No",
    "red_flags": "No",
    "risk_factors": "No"
  },
  "patient_id": "unknown"
}
```

Slot names can match your Lex intent. We build a transcript from them.

**Response:** Same as `POST /ai/triage`.

---

## 3. GET `/health`

**Use when:** Liveness / readiness checks.

**Response:** `{"ok": true}`

---

## Twilio Transcript Format

Twilio often sends transcripts like:
```
chief_complaint:I'm coughing. symptom_duration:1 month. symptom_severity:moderate. symptom_progression:better. aggravating_factors:Drinking alcohol. relieving_factors:Physical activity. associated_symptoms:No. red_flags:No. risk_factors:No
```

Send it as-is in the `transcript` field of `POST /ai/triage`. No preprocessing needed.

---

## Example Lambda Call (Python)

```python
import httpx

TRIAGE_URL = "http://3.146.34.91:8100"

def call_triage(transcript: str, patient_id: str = "unknown"):
    r = httpx.post(
        f"{TRIAGE_URL}/ai/triage",
        json={"transcript": transcript, "patient_id": patient_id},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()

# Or with slots
def call_triage_from_slots(slots: dict, patient_id: str = "unknown"):
    r = httpx.post(
        f"{TRIAGE_URL}/ai/triage/from-slots",
        json={"slots": slots, "patient_id": patient_id},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()
```
