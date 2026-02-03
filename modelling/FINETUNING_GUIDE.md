# LLM Finetuning Pipeline for ENT Triage

Finetune a language model on your ENT data for domain-specific accuracy. Designed to scale from local development to production.

## Quick Start

### 1. Prepare Data

```bash
cd modelling
python code/prepare_data.py
```

This converts your ENT datasets (CSV/XLSX) into training format:
- Input: `modelling/data/*.csv`, `modelling/data/*.xlsx`, nested folders
- Output: `modelling/data/training_data.jsonl`

Check the output format:
```bash
head -1 modelling/data/training_data.jsonl | jq .
```

### 2. Finetune Model

```bash
# Install dependencies
pip install -r modelling/requirements.txt

# Finetune (local, ~30-60 min on CPU, 5-15 min on GPU)
python modelling/code/finetune_unsloth.py \
    --model-name "qwen2.5-0.5b" \
    --data-file "modelling/data/training_data.jsonl" \
    --output-dir "modelling/model/finetuned-ent-llm" \
    --epochs 3 \
    --batch-size 4
```

**What this does:**
- Downloads base model (qwen2.5-0.5b)
- Trains on your ENT data with LoRA (parameter-efficient)
- Saves finetuned model to `modelling/model/finetuned-ent-llm/`

**Available models:**
- `qwen2.5-0.5b` (smallest, fastest, ~500MB)
- `qwen2-7b` (balanced)
- `phi-2` (lightweight)
- `mistral-7b` (larger, better quality)

Or pass any Hugging Face model ID directly.

### 3. Export to Ollama

```bash
python modelling/code/export_to_ollama.py \
    --model-dir "modelling/model/finetuned-ent-llm" \
    --ollama-model-name "ent-triage-qwen2.5"
```

Then manually create the Ollama model:
```bash
cd modelling/model/finetuned-ent-llm
ollama create ent-triage-qwen2.5 -f Modelfile
```

### 4. Update API

Edit `app/config.py`:
```python
OLLAMA_MODEL_NAME = "ent-triage-qwen2.5"  # Your finetuned model
```

Restart the API:
```bash
python -m uvicorn app.main:app --reload --port 8100
```

Test:
```bash
curl -X POST http://localhost:8100/ai/triage \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "PAT123",
    "transcript": "Patient has severe sore throat"
  }' | jq .
```

---

## Scaling to Production

### Local Development (Now)
✅ Unsloth on your machine
- Pros: Free, fast iteration, full control
- Cons: Limited by your hardware

### Small Scale (100-500k examples)
**AWS SageMaker** (recommended if you have budget)
```bash
# SageMaker handles scaling automatically
# Estimated cost: $0.30-2/hour training time
```

**Steps:**
1. Push training data to S3
2. Create SageMaker training job with HuggingFace estimator
3. Monitor training in AWS Console
4. Download finetuned model
5. Deploy to Ollama (same export step)

### Enterprise Scale (Millions of examples, daily retraining)
**Custom distributed training**
- EC2 multi-GPU setup with PyTorch Distributed Data Parallel (DDP)
- CI/CD pipeline to retrain on new data weekly
- A/B test models before deployment
- MLflow for experiment tracking

---

## Architecture: Scalable Design

Your current setup is built for scale:

```
Raw ENT Data
    ↓
prepare_data.py → training_data.jsonl (standardized format)
    ↓
finetune_unsloth.py (current: local)
    ↓
export_to_ollama.py → Ollama format
    ↓
app/ollama_client.py (unchanged!)
    ↓
API Response
```

**To swap training backend later (e.g., SageMaker):**
- Only replace `finetune_unsloth.py`
- `prepare_data.py` → `training_data.jsonl` stays the same
- `export_to_ollama.py` stays the same
- API code doesn't change

---

## Troubleshooting

### OOM (Out of Memory) on Local Machine

Reduce batch size or sequence length:
```bash
python modelling/code/finetune_unsloth.py \
    --batch-size 2 \  # Reduce from 4
    --model-name "qwen2.5-0.5b"  # Use smaller model
```

### Training too slow

Use GPU or smaller model:
```bash
# Check if GPU available
python -c "import torch; print(torch.cuda.is_available())"

# If no GPU, use Unsloth on Google Colab (free GPU)
```

### Model quality poor

- More training data needed (add more examples)
- Increase num_epochs: `--epochs 5`
- Better quality labels (clean up training data)
- Use larger base model: `--model-name "qwen2-7b"`

---

## Monitoring & Validation

After finetuning, evaluate on test set:

```python
# Create modelling/code/evaluate.py
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "modelling/model/finetuned-ent-llm"
model = AutoModelForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load test examples
test_data = load_dataset("json", data_files="modelling/data/test.jsonl")

# Evaluate...
```

---

## Next Steps

1. ✅ Run `prepare_data.py` and check output format
2. ✅ Finetune with `finetune_unsloth.py`
3. ✅ Export and deploy with `export_to_ollama.py`
4. ✅ Test API with finetuned model
5. ⏭️ Create evaluation script (for accuracy metrics)
6. ⏭️ Set up automated retraining pipeline (when you have new data)

---

## References

- **Unsloth:** https://github.com/unslothai/unsloth (4x-7x faster finetuning)
- **SageMaker Training:** https://docs.aws.amazon.com/sagemaker/latest/dg/hugging-face.html
- **LoRA:** https://arxiv.org/abs/2106.09685 (parameter-efficient finetuning)
- **Ollama Models:** https://github.com/ollama/ollama
