from fastapi import FastAPI
from app.routes import router
from app.config import settings

app = FastAPI(title="AI Triage Service")
app.include_router(router)

@app.get("/health")
def health():
    return {"ok": True}
