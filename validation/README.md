# Summary Validation Framework

This folder provides a **rule-based** validation workflow for the ENT triage LLM **summary** output. It scores each summary on three dimensions without using an LLM-as-judge.

## Dimensions

| Metric | What it measures |
|--------|-------------------|
| **Correctness** | Summary facts are consistent with the transcript: no contradiction of negations (e.g. transcript says "no fever", summary must not claim fever), and key facts (duration, severity) are reflected. |
| **Faithfulness** | Summary is grounded in the transcript: proportion of summary tokens that appear in the transcript (reduces hallucination). |
| **Relevance** | Summary is relevant to ENT triage: presence of symptom/severity/duration/urgency terms and reasonable length. |

All scores are in **0–1** (higher is better). Faithfulness uses strict token overlap, so paraphrased but accurate summaries may score lower until you add stemming or semantic similarity.

## Workflow

```mermaid
flowchart LR
  A[Generate or load eval data] --> B[Get summary per row]
  B --> C[Compute correctness, faithfulness, relevance]
  C --> D[Aggregate report]
```

1. **Generate synthetic eval data** (optional): transcripts + reference summaries + urgency.
2. **Get summary** for each transcript: either use the **reference summary** from the eval file, or call the **triage API** to get the live LLM summary.
3. **Run metrics**: for each (transcript, summary) pair, compute correctness, faithfulness, relevance.
4. **Report**: print per-sample and average scores.

## Setup

From the **ent_ai_triage** project root:

```bash
cd ent_ai_triage
source venv/bin/activate   # if you use a venv
pip install -r requirements.txt   # httpx only needed for --summary-source=api
```

No extra dependencies are required for the rule-based metrics. `httpx` is only needed if you use `--summary-source=api`.

## Commands

### 1. Generate synthetic eval data and run validation (reference summaries)

```bash
python -m validation.run_validation --generate --verbose
```

This writes `validation/data/synthetic_eval.jsonl` and runs validation using the **reference_summary** field in that file (gold summaries). Use this to sanity-check the metrics.

### 2. Run validation on existing eval file (reference summaries)

```bash
python -m validation.run_validation --eval-file validation/data/synthetic_eval.jsonl --verbose
```

### 3. Run validation using summaries from the triage API

Start the triage API (e.g. `uvicorn app.main:app --port 8100`), then:

```bash
python -m validation.run_validation --eval-file validation/data/synthetic_eval.jsonl --summary-source api --api-url http://localhost:8100 --verbose
```

This sends each transcript to `POST /ai/triage`, uses the returned **summary** field, and scores it against the transcript.

## Eval file format (JSONL)

Each line is a JSON object:

- **transcript** (str): Call or patient transcript.
- **reference_summary** (str): Gold summary (used when `--summary-source=reference`).
- **urgency** / **expected_urgency** (str): `routine` | `semi-urgent` | `urgent` (for reference; metrics do not score urgency here).

Example:

```json
{"transcript": "Patient has mild sore throat for 2 days. No fever. Improving with tea.", "reference_summary": "Patient reports throat symptoms. Mild severity. Improving trend. Duration documented. No fever. Urgency: routine.", "urgency": "routine", "expected_urgency": "routine"}
```

## Output

- **Correctness** (avg), **Faithfulness** (avg), **Relevance** (avg), and sample count.
- With `--verbose`: per-sample transcript/summary excerpts and the three scores.

## Files

| File | Purpose |
|------|--------|
| `metrics.py` | Rule-based scorers: `score_correctness`, `score_faithfulness`, `score_relevance`, and `validate_summary()`. |
| `synthetic_data.py` | Generates `data/synthetic_eval.jsonl` with (transcript, reference_summary, urgency). |
| `run_validation.py` | CLI: load eval data, get summaries (reference or API), run metrics, print report. |
| `data/synthetic_eval.jsonl` | Generated eval set (create with `--generate`). |

## Extending

- **Correctness**: Adjust negation/duration/severity patterns in `metrics.py` or add more fact checks.
- **Faithfulness**: Swap token overlap for NLI or embedding-based entailment if you add a model later.
- **Relevance**: Extend `ENT_SYMPTOM_TERMS` or add length/format rules.
- **LLM-as-judge**: Later you can add a separate step that calls an LLM to score the same dimensions and compare to these rule-based scores.
