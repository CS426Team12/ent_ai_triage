import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Triage Service")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every incoming request so you can see when calls go through (e.g. Twilio)."""
    start = time.time()
    client = request.client.host if request.client else "unknown"
    logger.info(f"CALL RECEIVED | {request.method} {request.url.path} | client={client}")
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(f"CALL COMPLETE | {request.method} {request.url.path} | status={response.status_code} | {elapsed:.2f}s")
    return response
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
def health():
    return {"ok": True, "status": "running"}
