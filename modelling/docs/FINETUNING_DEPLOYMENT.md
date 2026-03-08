# Finetuned Model: Deployment & Hosting

After finetuning your ENT triage model, you need to host it and configure the AI backend to call it. Groq does **not** host custom models. Options:

**Config:** Copy `ent_ai_triage/.env.example` to `.env` and set `LLM_PROVIDER`, model URL, and keys for your chosen host.

---

## 1. Ollama (Local or EC2)

**Best for:** Local dev, EC2 self-hosted, low latency.

### Step 1: Convert to GGUF and create Ollama model

After finetuning (Colab/notebook saves to `./triage_qwen_lora` or `modelling/model/finetuned-ent-llm`):

```bash
# Merge LoRA into base model (if you used LoRA)
# In Python: merged = model.merge_and_unload(); merged.save_pretrained("./triage_qwen_merged")

# Convert to GGUF (use llama.cpp or transformers)
pip install gguf
# Or use: https://github.com/ggerganov/llama.cpp for conversion

# Create Ollama model
ollama create ent-triage-qwen -f Modelfile
```

**Modelfile** (update the `FROM` path to your merged model):

```
FROM ./triage_qwen_merged

PARAMETER temperature 0.3
PARAMETER num_predict 1024

SYSTEM """You are an ENT triage clinician. Format response as:
SUMMARY: [at least 3 sentences]
FINDINGS: ...
FLAGS: [TAG] keyword, ...
URGENCY: routine|semi-urgent|urgent
REASONING: ...
"""
```

### Step 2: Run Ollama

```bash
ollama serve
ollama run ent-triage-qwen
```

### Step 3: Configure AI backend

In `ent_ai_triage/.env`:

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=ent-triage-qwen
```

(For EC2, use `OLLAMA_BASE_URL=http://EC2_IP:11434` if Ollama runs on another host.)

---

## 2. Together.ai (Hosted)

**Best for:** Hosted inference, no server management.

1. Push your merged model to Hugging Face.
2. Add the model to Together.ai (dashboard or API).
3. Add a Together client in `ollama_client.py` (similar to Groq) that calls:
   - `https://api.together.xyz/v1/chat/completions` with your model name.

4. In `.env`:
   ```
   LLM_PROVIDER=together
   TOGETHER_API_KEY=...
   TOGETHER_MODEL=your-org/ent-triage-qwen
   ```

---

## 3. Replicate

**Best for:** Simple hosted inference.

1. Package your model as a Replicate model (Docker + Cog).
2. Use the Replicate API to run inference.
3. Add a `_call_replicate()` function in `ollama_client.py` and wire it when `LLM_PROVIDER=replicate`.

---

## 4. EC2 Self-Hosted Ollama

Same as option 1, but run Ollama on EC2:

1. Install Ollama on EC2.
2. Convert and create your model.
3. Run `ollama serve` (or use systemd).
4. AI backend `.env`: `OLLAMA_BASE_URL=http://EC2_IP:11434`
5. Security group: open port 11434.

---

## Summary

| Hosting | Setup | AI Backend Config |
|---------|-------|-------------------|
| **Ollama (local)** | `ollama serve`, create model | `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://localhost:11434` |
| **Ollama (EC2)** | Ollama on EC2, port 11434 | `OLLAMA_BASE_URL=http://EC2_IP:11434` |
| **Together.ai** | Upload model, get API key | Add Together client, `LLM_PROVIDER=together` |
| **Replicate** | Package as Replicate model | Add Replicate client, `LLM_PROVIDER=replicate` |

The `ollama_client.py` already supports Ollama; switch `LLM_PROVIDER` to `ollama` and set `OLLAMA_MODEL_NAME` to your finetuned model name.

---

## Verify after deployment

1. Restart the AI backend: `uvicorn app.main:app --host 0.0.0.0 --port 8100` (from `ent_ai_triage/`).
2. Run the pipeline test: `python scripts/run_local_pipeline_test.py` (from `ent_ai_triage/`).
3. Optional: `curl -X POST http://localhost:8100/ai/test-pipeline` and check the JSON response for `backend_saved` and triage summary.
