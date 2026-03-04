# Unit tests for ENT AI Triage

## Layout

- **`conftest.py`** – Pytest fixtures, env setup for Settings, FastAPI `TestClient`.
- **`test_main.py`** – FastAPI app, `GET /health`, middleware.
- **`test_routes.py`** – Triage endpoints (`/ai/triage`, `/ai/triage/from-slots`), request/response models, `_build_transcript_from_slots`, `_transcript_preview`.
- **`test_config.py`** – Settings, `get_settings`, `SQLALCHEMY_DATABASE_URL`.
- **`test_ollama_client.py`** – `parse_triage_response`, `parse_flags`, `extract_flags_from_transcript`, `validate_urgency_classification`.
- **`test_backend_client.py`** – `_is_known_patient`, `get_patient_history`, `save_triage_to_backend` (mocked).
- **`test_ml_client.py`** – Feature extraction helpers, `predict_urgency` (mocked when no model).
- **`test_utils.py`** – `extract_json_from_model_output`.

## Run

From project root:

```bash
cd ent_ai_triage
pip install -r requirements.txt
pytest tests/ -v
```

Run a single file:

```bash
pytest tests/test_routes.py -v
```

## Env

Tests set minimal env in `conftest.py` so `Settings()` can load: `BACKEND_BASE_URL`, `BACKEND_USERNAME`, `BACKEND_PASSWORD`, `OLLAMA_BASE_URL`. No real backend or Ollama is required; external calls are mocked in route and backend tests.
