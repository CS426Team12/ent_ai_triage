# from fastapi import APIRouter
# from pydantic import BaseModel
# from app.ollama_client import call_ollama

# router = APIRouter(prefix="/ai")

# class TriageRequest(BaseModel):
#     transcript: str

# class TriageResponse(BaseModel):
#     summary: str
#     urgency: str

# @router.post("/triage", response_model=TriageResponse)
# async def triage(payload: TriageRequest):
#     transcript = payload.transcript

#     # Call tiny model (raw text output)
#     raw = await call_ollama(transcript)

#     # Minimal processing — tiny model can't follow big prompts
#     summary = raw.strip()[:500]
#     urgency = "unknown"

#     return TriageResponse(summary=summary, urgency=urgency)


from fastapi import APIRouter
from pydantic import BaseModel
from app.ollama_client import call_ollama
from app.utils import extract_json_from_model_output
from app.backend_client import save_triage_to_backend

router = APIRouter(prefix="/ai")

class TriageRequest(BaseModel):
    transcript: str
    patient_id: str  # YOU NEED THIS FOR SAVING INTO DB

class TriageResponse(BaseModel):
    summary: str
    urgency: str

@router.post("/triage", response_model=TriageResponse)
async def triage(payload: TriageRequest):
    # 1. Run LLM
    raw = await call_ollama(payload.transcript)

    # 2. Parse model response
    result = extract_json_from_model_output(raw)
    summary = result.get("summary", "")
    urgency = result.get("urgency", "unknown")

    # 3. Save to backend DB
    await save_triage_to_backend(payload.patient_id, summary, urgency)

    # 4. Return response to UI
    return TriageResponse(summary=summary, urgency=urgency)
