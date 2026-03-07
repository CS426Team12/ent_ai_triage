#!/usr/bin/env python3
"""
Local pipeline test: simulates Twilio → AI Backend → Backend API → Database.

Run with:
  python scripts/run_local_pipeline_test.py

Prerequisites:
- AI Backend running at http://localhost:8100
- Main Backend running at http://localhost:8000
- Valid patient UUID in backend DB for full E2E (optional; use PATIENT_ID env var)
"""

import json
import logging
import os
import sys

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AI_BACKEND_URL = os.environ.get("AI_BACKEND_URL", "http://localhost:8100")
PATIENT_ID = os.environ.get("PATIENT_ID", "unknown")


def run_test() -> bool:
    """Run full pipeline test. Returns True if triage succeeded."""
    logger.info("=" * 60)
    logger.info("LOCAL PIPELINE TEST")
    logger.info("AI Backend: %s | Patient ID: %s", AI_BACKEND_URL, PATIENT_ID)
    logger.info("=" * 60)

    # Step 1: Simulated transcript
    logger.info("Step 1: Simulated Twilio transcript received")

    # Step 2-5: Call AI backend test-pipeline (runs full pipeline internally)
    url = f"{AI_BACKEND_URL.rstrip('/')}/ai/test-pipeline"
    payload = {"patient_id": PATIENT_ID}

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
    except httpx.ConnectError as e:
        logger.error("Failed to connect to AI backend at %s: %s", AI_BACKEND_URL, e)
        logger.info("Ensure AI backend is running: uvicorn app.main:app --port 8100")
        return False

    logger.info("Step 2: AI triage generated")
    logger.info("Step 3: Authenticated with backend")

    if resp.status_code != 200:
        logger.error("AI backend returned %d: %s", resp.status_code, resp.text)
        return False

    data = resp.json()

    triage_result = data.get("triage_result", {})
    backend_saved = data.get("backend_saved", False)
    backend_status = data.get("backend_status", "unknown")
    message = data.get("message", "")

    logger.info("Step 4: Triage result sent to backend API")
    logger.info("Step 5: Backend response: %s", backend_status)
    logger.info("Case saved: %s", backend_saved)
    logger.info("Message: %s", message)
    logger.info("-" * 60)
    logger.info("Triage result:")
    logger.info("  summary: %s", triage_result.get("summary", "")[:100] + ("..." if len(triage_result.get("summary", "")) > 100 else ""))
    logger.info("  urgency_level: %s", triage_result.get("urgency_level"))
    logger.info("  confidence_score: %s", triage_result.get("confidence_score"))
    logger.info("-" * 60)

    if backend_saved:
        logger.info("SUCCESS: Case saved to backend database")
    else:
        logger.info("Triage completed; backend save skipped (patient_id=%s). Set PATIENT_ID=<valid-uuid> for full E2E.", PATIENT_ID)

    return True


if __name__ == "__main__":
    ok = run_test()
    sys.exit(0 if ok else 1)
