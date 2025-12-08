# import httpx
# from app.config import settings

# async def call_ollama(prompt: str) -> str:
#     """
#     Sends a prompt to the Ollama server and returns the full concatenated response.
#     Works with streamed models like Qwen2.5:0.5b.
#     """
#     url = f"{settings.OLLAMA_BASE_URL}/api/generate"

#     async with httpx.AsyncClient(timeout=60.0) as client:
#         response = await client.post(
#             url,
#             json={
#                 "model": settings.OLLAMA_MODEL_NAME,
#                 "prompt": prompt,
#                 "stream": True
#             }
#         )

#         response.raise_for_status()

#         full_text = ""

#         async for chunk in response.aiter_lines():
#             if not chunk.strip():
#                 continue
#             try:
#                 obj = httpx.Response.json(httpx.Response(200, content=chunk))
#                 if "response" in obj:
#                     full_text += obj["response"]
#             except:
#                 continue

#         return full_text



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

    async with httpx.AsyncClient() as client:
        headers = {"Content-Type": "application/json"}

        resp = await client.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            headers=headers
        )

    # if resp.status_code >= 400:
    #     print("❌ Backend /triage-cases error:", resp.status_code)
    #     print("❌ Error body:", resp.text)   # <— ADD THIS LINE
    #     print("❌ Sent payload:", payload)    # <— AND THIS LINE

    resp.raise_for_status()

    data = resp.json()
    raw_text = data.get("response", "")

    return raw_text
