# ENT AI Triage System

A production-ready AI-powered ENT patient triage system that prioritizes patients based on symptom severity, medical history, and clinical urgency indicators.

**Key Feature:** Explainable urgency classification with transparent reasoning and tagged clinical flags.

## Overview

This system provides:

1. **LLM-Based Triage** – Uses Ollama (local LLM) for clinical analysis
2. **Medical History Integration** – Considers patient's medical background for context-aware decisions
3. **Multi-Layer Urgency Validation** – Conservative escalation approach with critical red-flag detection
4. **Transparent Reasoning** – Every classification includes flags and justification
5. **AWS Integration** – Connect/Lex/Lambda pipeline for real-time patient call processing

## Architecture

```
Patient Call (Phone)
    ↓
AWS Connect (Call Routing)
    ↓
Lex Bot (NLU + Conversation)
    ↓
Lambda Function (Triage Caller)
    ↓
Your Triage API (FastAPI on EC2)
    ├── LLM Analysis (Ollama)
    ├── Medical History Lookup
    ├── Multi-Layer Validation
    └── Flag Extraction
    ↓
Urgency Classification + Reasoning
```

## Quick Start

### 1. Local Development

```bash
# Clone and setup
git clone https://github.com/joshmatni/ent_ai_triage.git
cd ent_ai_triage
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run API locally
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8100
```

### 2. Test the API

```bash
# In another terminal
curl -X POST http://localhost:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT123",
    "transcript": "Patient reports severe sore throat for 3 days, worsening. Has difficulty swallowing."
  }'
```

### 3. Deploy to Production

See `docs/EC2_DEPLOYMENT_GUIDE.md` for full deployment instructions.

## Core Components

### `app/routes.py`
**Endpoint:** `POST /ai/triage`

**Request:**
```json
{
  "patient_id": "PAT123",
  "transcript": "Patient call transcript"
}
```

**Response:**
```json
{
  "summary": "Clinical summary of findings",
  "urgency": "urgent|semi-urgent|routine",
  "findings": ["Key finding 1", "Key finding 2"],
  "flags": [
    {"tag": "RED_FLAG", "keyword": "breathing difficulty"},
    {"tag": "SEVERITY", "keyword": "severe"},
    {"tag": "MEDICAL_HISTORY", "keyword": "immunocompromised"}
  ],
  "reasoning": "Why this urgency level was assigned",
  "ml_confidence": 0.95,
  "urgency_confidence": "high|medium|low"
}
```

### `app/ollama_client.py`
Handles LLM integration with:
- **call_ollama()** – Sends transcript + medical history to Ollama
- **parse_triage_response()** – Extracts structured fields from LLM output
- **extract_flags_from_transcript()** – Fallback keyword-based flag extraction
- **validate_urgency_classification()** – Multi-layer validation with medical history escalation

### `app/backend_client.py`
Fetches patient medical history:
- `get_patient_history(patient_id)` – Retrieves medical history, allergies, previous visits
- `save_triage_to_backend()` – Persists triage results

### `app/ml_client.py`
Secondary ML model for confidence scoring:
- Scikit-learn RandomForest + DecisionTree ensemble
- Provides `ml_confidence` score alongside LLM urgency

## Urgency Classification Logic

### Decision Rules (Priority Order)

1. **Critical Red Flags Detected** → **URGENT** (high confidence)
   - Breathing difficulty, stridor, wheezing
   - Severe pain, dysphagia affecting airway
   - Sudden hearing loss, severe dizziness
   - Immunocompromised + infection signs
   - Fever + severe localized symptoms

2. **Immunocompromised Patient** → Escalate one level
   - HIV/AIDS, cancer, on immunosuppressants
   - Any infection signs → SEMI-URGENT minimum

3. **High-Risk Patient + Worsening** → **SEMI-URGENT**
   - Diabetes, chronic lung disease, heart disease
   - Symptoms deteriorating in hours/days

4. **Mild + Stable + No Red Flags** → **ROUTINE**
   - Improving trend, no dangerous symptoms

### Conservative Approach

When in doubt → **Escalate** (favor false positives over false negatives)

## Medical History Integration

Patient medical history automatically adjusts triage urgency:

- **Immunocompromised** – Any symptoms escalated
- **Diabetes** – Increased infection risk considered
- **Chronic Lung Disease** – Any breathing changes = urgent
- **Previous ENT Complications** – Escalated one level
- **Antibiotic Allergies** – Noted for clinical team

## Flag Categories

Triage flags are tagged with clinical meaning:

| Tag | Meaning | Examples |
|-----|---------|----------|
| `SYMPTOM` | Main ENT symptom | sore throat, congestion |
| `SEVERITY` | Intensity level | mild, moderate, severe |
| `PROGRESSION` | Trend direction | improving, worsening, stable |
| `RED_FLAG` | Critical danger sign | breathing difficulty, hearing loss |
| `MEDICAL_HISTORY` | Relevant past condition | diabetes, immunocompromised |
| `DURATION` | Symptom length | 2 days, 1 week |
| `ASSOCIATED_SYMPTOMS` | Secondary symptoms | headache, fever, fatigue |
| `RELIEVING_FACTORS` | What helps | rest, warm tea, medication |
| `AGGRAVATING_FACTORS` | What worsens | swallowing, talking |

## Deployment

### AWS Lambda

Deploy the triage caller Lambda function:

```bash
# Option 1: Automated (recommended)
export TRIAGE_API_URL=http://your-ec2-ip:8100
python3 deploy_lambda.py

# Option 2: Manual
aws lambda create-function \
  --function-name ent-ai-triage \
  --runtime python3.12 \
  --handler aws_lambda_handler.lambda_handler \
  --zip-file fileb://lambda_package.zip
```

See `docs/AWS_CONNECT_LEX_GUIDE.md` for full Lex/Connect integration.

### EC2 Hosting

Deploy your FastAPI backend to EC2:

```bash
# SSH into EC2 instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Follow EC2_DEPLOYMENT_GUIDE.md for full setup
```

See `docs/EC2_DEPLOYMENT_GUIDE.md` for detailed instructions.

## Configuration

### Environment Variables (`.env`)

```
# Ollama LLM
OLLAMA_BASE_URL=http://18.224.183.103:11434
OLLAMA_MODEL_NAME=qwen2.5:0.5b

# Backend API (for patient history)
BACKEND_BASE_URL=http://localhost:8000
BACKEND_USERNAME=your_username
BACKEND_PASSWORD=your_password

# Redis (optional)
AI_REDIS_URL=redis://localhost:6379

# API Port
PORT=8100
```

### ML Model

The system uses a pre-trained RandomForest model from `modelling/model/final_ent_triage_model.pkl` for secondary confidence scoring.

## Directory Structure

```
ent_ai_triage/
├── README.md (this file)
├── docs/
│   ├── AWS_CONNECT_LEX_GUIDE.md      # AWS integration guide
│   ├── EC2_DEPLOYMENT_GUIDE.md       # EC2 deployment steps
│   └── ARCHITECTURE_UPDATE.md        # Architecture notes
├── app/
│   ├── main.py                       # FastAPI app
│   ├── routes.py                     # /ai/triage endpoint
│   ├── ollama_client.py              # LLM integration
│   ├── backend_client.py             # Patient history fetching
│   ├── ml_client.py                  # ML model scoring
│   ├── prompts.py                    # LLM system prompts
│   ├── schemas.py                    # Pydantic models
│   ├── config.py                     # Configuration
│   └── utils.py                      # Utilities
├── modelling/
│   ├── code/
│   │   ├── preprocessing.py
│   │   └── models/
│   │       ├── logistic_regression.py
│   │       ├── naive_bayes.py
│   │       └── test_random_forest.py
│   └── model/
│       └── final_ent_triage_model.pkl
├── aws_lambda_handler.py             # Lambda function
├── deploy_lambda.py                  # Lambda deployment script
├── deploy_lambda.sh                  # Lambda deployment (Bash)
└── requirements.txt
```

## API Examples

### Example 1: Routine Case

```bash
curl -X POST http://localhost:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT001",
    "transcript": "Patient has mild sore throat for 2 days. Feeling better with rest and tea. No fever."
  }'
```

**Response:**
```json
{
  "summary": "Patient with mild pharyngitis improving with conservative measures.",
  "urgency": "routine",
  "findings": ["Mild sore throat", "Improving trend"],
  "flags": [
    {"tag": "SYMPTOM", "keyword": "sore throat"},
    {"tag": "SEVERITY", "keyword": "mild"},
    {"tag": "PROGRESSION", "keyword": "improving"}
  ],
  "reasoning": "Mild symptoms with improving trend, no red flags.",
  "ml_confidence": 0.92,
  "urgency_confidence": "high"
}
```

### Example 2: Urgent Case

```bash
curl -X POST http://localhost:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT002",
    "transcript": "Patient has severe sore throat, difficulty swallowing, fever 101F. Immunocompromised (HIV). Symptoms worsening over 24 hours."
  }'
```

**Response:**
```json
{
  "summary": "Immunocompromised patient with severe pharyngitis and fever requiring urgent evaluation.",
  "urgency": "urgent",
  "findings": [
    "Severe sore throat",
    "Dysphagia",
    "Fever (101F)",
    "Rapid deterioration",
    "Immunocompromised status"
  ],
  "flags": [
    {"tag": "RED_FLAG", "keyword": "difficulty swallowing"},
    {"tag": "SEVERITY", "keyword": "severe"},
    {"tag": "RED_FLAG", "keyword": "fever"},
    {"tag": "MEDICAL_HISTORY", "keyword": "HIV"},
    {"tag": "PROGRESSION", "keyword": "worsening"}
  ],
  "reasoning": "Immunocompromised with severe symptoms + fever = urgent. Potential airway compromise risk.",
  "ml_confidence": 0.96,
  "urgency_confidence": "high"
}
```

## Development

### Adding New Red Flags

Edit `app/ollama_client.py`:

```python
CRITICAL_RED_FLAGS = [
    "breathing difficulty",
    "difficulty breathing",
    # Add new flags here
    "your_new_flag"
]
```

### Extending Medical History Rules

Edit `validate_urgency_classification()` in `app/ollama_client.py`:

```python
# Add to MEDICAL_RISK_FACTORS
"your_condition": ["keyword1", "keyword2"],
```

## Performance Considerations

- **Ollama Timeout:** 120 seconds (configurable in `ollama_client.py`)
- **Lambda Timeout:** 60 seconds (set in AWS Lambda configuration)
- **ML Model Load:** ~500ms on first request, cached thereafter
- **API Response Time:** 1-5 seconds typical (depends on LLM)

## Monitoring & Logging

Logs include:
- ✓ Patient history fetch status
- 📋 LLM urgency classification
- 🚩 Flags detected count
- 🤖 ML confidence score
- ✓ Urgency validation adjustments

**View EC2 logs:**
```bash
sudo journalctl -u ent-triage -f
```

## Team Responsibilities

- **You:** Triage API backend (FastAPI), Lambda triage-caller, EC2 deployment
- **Teammate:** Lex bot, Connect contact flow, patient data collection

## Support & Documentation

- **Architecture:** See `docs/ARCHITECTURE_UPDATE.md`
- **AWS Integration:** See `docs/AWS_CONNECT_LEX_GUIDE.md`
- **EC2 Deployment:** See `docs/EC2_DEPLOYMENT_GUIDE.md`

## Authors

- [@joshmatni](https://github.com/joshmatni)
- [@KBTrotter](https://github.com/KBTrotter)
- [@ployw](https://github.com/ployw)
- [@Calingo-Angelo](https://github.com/Calingo-Angelo)

## License

[Your License Here]
