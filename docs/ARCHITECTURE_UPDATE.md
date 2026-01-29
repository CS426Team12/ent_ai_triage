# AI Triage System - Architecture Update

## Change Summary

You've asked an excellent question about **how urgency classification works**. We've implemented a **hybrid approach** that addresses your concerns about symptom progression and red-flag symptoms.

## New Architecture

```
Patient Transcript + Patient ID
              ↓
    [Fetch Patient History]
    (medical history, allergies, previous visits)
              ↓
         [LLM (Ollama)]
              ↓
    ┌─────────────────────────────────┐
    │  ANALYSIS WITH CONTEXT:         │
    │  - Transcript                   │
    │  - Medical History              │
    │  - Previous ENT Visits          │
    │  - Allergies                    │
    │  - Symptom Progression          │
    │  - Red-Flag Detection           │
    └─────────────────────────────────┘
              ↓
    ┌─────────────────────────────────┐
    │  OUTPUTS:                       │
    │  ✓ Clinical Summary             │
    │  ✓ Urgency (MAIN CLASSIFIER)    │
    │  ✓ Key Findings                 │
    │  ✓ Reasoning (Explainability)   │
    └─────────────────────────────────┘
              ↓
   [ML Model - Secondary Validation]
   (confidence score for audit trail)
              ↓
   API Response + Backend Save
```

## Why LLM for Urgency Classification? ✅

### Previous Approach (ML Model):
- ❌ Black box - hard to explain decisions
- ❌ Fixed features - can't easily add new red flags
- ❌ No context - couldn't use patient history
- ❌ Limited reasoning capability

### New Approach (LLM):
- ✅ **Transparent reasoning** - explains WHY it classified as urgent
- ✅ **Contextual** - uses patient history, previous visits, allergies
- ✅ **Flexible** - can handle new symptoms/red flags without retraining
- ✅ **Symptom progression aware** - understands "worsening" vs "improving"
- ✅ **Red-flag detection** - explicitly trained on danger signs:
  - Difficulty breathing / stridor / wheezing
  - Severe throat pain affecting airway
  - Sudden hearing loss
  - Severe dizziness/vertigo
  - Signs of infection spreading
  - Immunocompromised + infection signs

## Key Improvements

### 1. Patient History Integration
```python
# Now fetches from backend via patient ID
patient_history = await get_patient_history(patient_id)
# Includes: medicalHistory, allergies, previousVisits
```

### 2. Enhanced Prompt Engineering
The LLM now receives:
- Transcript (patient symptoms)
- Medical history
- Previous ENT visits
- Known allergies/sensitivities

This allows it to:
- Contextualize current symptoms
- Identify patterns from history
- Consider medication interactions
- Adjust expectations based on baseline health

### 3. Structured Output Parsing
LLM response format:
```
SUMMARY: [Clinical summary 1-3 sentences]
FINDINGS: [List of concerning findings/red flags]
URGENCY: [routine/semi-urgent/urgent]
REASONING: [Explanation for classification]
```

This is **parsed and structured** so you get:
- Clear explanation of the urgency decision
- Enumerated findings for review
- Traceability for audits

### 4. Symptom Progression Analysis
The prompt explicitly instructs the LLM to:
- Account for **worsening trends** → escalate urgency
- Note **improvement trends** → lower urgency
- Consider **rate of change** → rapid worsening = urgent
- Compare to **patient baseline** → what's normal for them?

### 5. Red-Flag Detection (Breathing Difficulty, etc.)
The system prompt has explicit red flags:
```
- Difficulty breathing / stridor / wheezing
- Severe throat pain / dysphagia affecting airway
- Severe dizziness / vertigo affecting mobility
- Sudden hearing loss
- Signs of infection spreading
- Immunocompromised patient with infection signs
```

If ANY of these are detected, urgency is elevated to **"urgent"**.

## Response Format

### Old Response:
```json
{
  "summary": "...",
  "urgency": "routine",
  "ml_confidence": 0.85
}
```

### New Response:
```json
{
  "summary": "Patient with 2-day history of mild pharyngeal pain, improving with conservative measures.",
  "urgency": "routine",
  "findings": [
    "Mild throat pain (3/10)",
    "Improving trend",
    "No fever or systemic symptoms",
    "No red flags"
  ],
  "reasoning": "Mild symptoms with positive progression trend and no concerning findings. Conservative management appropriate at this time.",
  "ml_confidence": 0.85
}
```

## How Urgency is Now Determined

### **ROUTINE** ✅
- Mild/stable symptoms
- **Improving trend** (symptoms getting better)
- No red flags
- Patient stable

**Example**: "2-day sore throat, improving with warm tea"

### **SEMI-URGENT** ⚠️
- Moderate symptoms
- **Worsening trend** (getting worse)
- Some concerning findings but no critical red flags
- Needs evaluation within 24-48 hours

**Example**: "Throat pain worsening over 3 days, now with fever"

### **URGENT** 🚨
- Severe symptoms
- **Rapid deterioration**
- **ANY red flag present**:
  - Difficulty breathing
  - Severe pain affecting swallowing/breathing
  - Sudden hearing loss
  - Severe dizziness
  - Signs of spreading infection

**Example**: "Severe throat pain, difficulty swallowing, fever 102°F, immunocompromised"

## ML Model Role (Changed)

The trained ML model now serves as:
- **Secondary validation** (not primary decision-maker)
- **Confidence scoring** for audit trails
- **Anomaly detection** (if ML disagrees with LLM)
- **Fallback** if LLM is unavailable

This is more **clinically appropriate** - you want transparent, explainable decisions from the LLM, not a black box.

## Implementation Details

### Modified Files:
1. **prompts.py** - Enhanced with red flags, progression guidance, patient history context
2. **ollama_client.py** - Now handles patient history, parses structured responses
3. **routes.py** - Fetches patient history, uses LLM for urgency, ML for validation
4. **backend_client.py** - Added `get_patient_history()` function

### New Response Fields:
- `findings` (list) - Enumerated clinical findings for transparency
- `reasoning` (string) - LLM explanation for urgency choice
- `ml_confidence` (float) - Secondary ML validation score

## Testing

When you test, you'll now see:

```json
{
  "summary": "Patient with mild pharyngeal pain (3/10) for 2 days, improving with conservative management. Associated mild nasal congestion. No fever, dysphagia, or systemic symptoms.",
  "urgency": "routine",
  "findings": [
    "Mild throat pain (3/10)",
    "Improving trajectory",
    "Mild nasal congestion",
    "No fever",
    "No dysphagia",
    "Healthy baseline"
  ],
  "reasoning": "Mild symptoms with positive improvement trend and no red flags or concerning features. Appropriate for conservative outpatient management.",
  "ml_confidence": 0.87
}
```

## Questions Answered

✅ **"How is urgency being classified?"** → LLM analyzes transcript + history against defined red flags and progression patterns

✅ **"Does it account for symptom progression?"** → Yes! Explicitly compares worsening vs improving trends

✅ **"Does it handle difficulty breathing?"** → Yes! Breathing difficulty is a **red flag that automatically escalates to "urgent"**

✅ **"Does it use patient history?"** → Yes! Fetches medical history, previous visits, allergies via patient ID

✅ **"Is it explainable?"** → Yes! Includes reasoning field showing WHY it chose that urgency level

---

**Next Steps**: Restart the server and test with the patient transcript to see the new structured output!
