# Using Groq (Hosted LLM) Instead of Ollama

When Ollama cannot run on EC2 (e.g. disk space limits), use **Groq** for LLM triage. Groq offers a free tier and fast inference.

## Steps

### 1. Get a Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up
3. **API Keys** → Create API Key
4. Copy the key (e.g. `gsk_...`)

### 2. Update `.env` on EC2

Add or update these lines in `~/ent_ai_triage/.env`:

```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

Optional: change `GROQ_MODEL` to another supported model (e.g. `llama-3.1-70b-versatile`, `mixtral-8x7b-32768`).

### 3. Restart the FastAPI service

```bash
sudo systemctl restart ai-triage
```

### 4. Test

```bash
curl -X POST http://localhost:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{"transcript": "chief_complaint:I'\''m coughing. symptom_duration:1 month. symptom_severity:moderate", "patient_id": "unknown"}'
```

---

## Switching Back to Ollama Later

When you have Ollama running (bigger disk, new instance, or remote server):

1. Set `LLM_PROVIDER=ollama` in `.env`
2. Set `OLLAMA_BASE_URL` and `OLLAMA_MODEL_NAME`
3. Restart: `sudo systemctl restart ai-triage`

---

## Using Your Own Finetuned Model Later

- **Groq / Together / etc.** – Deploy your finetuned model to a provider that supports custom models. Set `GROQ_MODEL` (or the provider’s model name) to your model and add provider-specific config if needed.
- **Ollama** – Load your finetuned model in Ollama, set `OLLAMA_MODEL_NAME` to it, and switch `LLM_PROVIDER` back to `ollama`.
