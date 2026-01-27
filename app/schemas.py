class TriageRequest(BaseModel):
    transcript: str
    patient_id: str

class TriageResponse(BaseModel):
    summary: str
    urgency: str