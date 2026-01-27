import httpx
from app.config import settings
from app.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT_TEMPLATE

async def call_ollama(transcript: str) -> str:
    prompt = (
        TRIAGE_SYSTEM_PROMPT.strip() +
        "\n" +
        TRIAGE_USER_PROMPT_TEMPLATE.replace("<<TRANSCRIPT>>", transcript) ## changed from strip 
    )

    payload = {
        "model": settings.OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {"Content-Type": "application/json"}

            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                headers=headers
            )

        resp.raise_for_status()

        data = resp.json()
        raw_text = data.get("response", "")

        return raw_text
    
    except httpx.TimeoutException:
        print("⚠️ Ollama timeout - returning default summary")
        return "Patient presents with ENT-related symptoms. Further assessment needed."
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return f"Error calling Ollama: {str(e)}"
