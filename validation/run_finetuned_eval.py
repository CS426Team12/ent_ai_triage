"""
Evaluate the finetuned triage Ollama model on combined_triage_training.jsonl.

Computes correctness, faithfulness, relevance (on summary) and urgency accuracy.
Uses prompts from app/prompts.py via call_ollama. Logs progress to terminal.

Run from ent_ai_triage directory:
  python -m validation.run_finetuned_eval
  python -m validation.run_finetuned_eval --limit 20
  python -m validation.run_finetuned_eval --n-routine 100 --n-semi 50 --n-urgent 50 --verbose

Requires: Ollama running with the model in OLLAMA_MODEL_NAME (default triage-mistral; ollama list).
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Allow imports from ent_ai_triage
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.ollama_client import call_ollama
from app.config import settings
from validation.metrics import validate_summary


def parse_urgency_from_output(output: str) -> str | None:
    """Extract URGENCY from training output. Returns routine|semi-urgent|urgent or None."""
    m = re.search(r"URGENCY:\s*(\w+(?:-\w+)?)", output, re.IGNORECASE)
    if not m:
        return None
    u = m.group(1).lower()
    return u if u in ("routine", "semi-urgent", "urgent") else None


def load_gold_set(
    data_path: Path,
    n_routine: int = 50,
    n_semi: int = 30,
    n_urgent: int = 30,
) -> list[dict]:
    """Load and sample gold examples. Returns list of {transcript, expected_urgency}."""
    gold_by_urgency = defaultdict(list)
    seen = set()
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            transcript = row.get("transcript", "")
            output = row.get("output", "")
            expected = parse_urgency_from_output(output)
            if not expected or not transcript:
                continue
            key = (transcript, expected)
            if key in seen:
                continue
            seen.add(key)
            gold_by_urgency[expected].append({
                "transcript": transcript,
                "expected_urgency": expected,
            })
    routine = gold_by_urgency["routine"][:n_routine]
    semi = gold_by_urgency["semi-urgent"][:n_semi]
    urgent = gold_by_urgency["urgent"][:n_urgent]
    return routine + semi + urgent


async def run_one(gold: dict) -> dict:
    """Run one example: call Ollama, compute metrics, compare urgency."""
    transcript = gold["transcript"]
    expected = gold["expected_urgency"]
    patient_history = {}

    llm_result = await call_ollama(transcript, patient_history)
    model_summary = llm_result.get("summary", "")
    model_urgency = llm_result.get("urgency", "routine")

    scores = validate_summary(transcript, model_summary)
    urgency_match = model_urgency == expected

    return {
        "transcript_preview": transcript[:80] + "..." if len(transcript) > 80 else transcript,
        "expected_urgency": expected,
        "model_urgency": model_urgency,
        "model_summary": model_summary,
        "correctness": scores.correctness,
        "faithfulness": scores.faithfulness,
        "relevance": scores.relevance,
        "urgency_match": urgency_match,
    }


def print_report(results: list[dict], verbose: bool = False) -> None:
    """Print aggregate metrics and per-sample details."""
    # Always print the summary block (even when empty)
    print("\n", flush=True)
    print("=" * 60, flush=True)
    print("FINAL SUMMARY - AVG METRICS", flush=True)
    print("=" * 60, flush=True)

    if not results:
        print("  Accuracy:     N/A (no samples run)", flush=True)
        print("  Correctness:  N/A", flush=True)
        print("  Faithfulness: N/A", flush=True)
        print("  Relevance:    N/A", flush=True)
        print("  N samples:    0", flush=True)
        print("=" * 60, flush=True)
        sys.stdout.flush()
        return

    n = len(results)
    avg_c = sum(r["correctness"] for r in results) / n
    avg_f = sum(r["faithfulness"] for r in results) / n
    avg_r = sum(r["relevance"] for r in results) / n
    accuracy = sum(1 for r in results if r["urgency_match"]) / n
    n_correct = sum(1 for r in results if r["urgency_match"])

    print(f"  Accuracy:     {accuracy:.1%} ({n_correct}/{n})", flush=True)
    print(f"  Correctness:  {avg_c:.3f}", flush=True)
    print(f"  Faithfulness: {avg_f:.3f}", flush=True)
    print(f"  Relevance:    {avg_r:.3f}", flush=True)
    print(f"  N samples:    {n}", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()

    # Per-urgency breakdown
    by_expected = defaultdict(list)
    for r in results:
        by_expected[r["expected_urgency"]].append(r)

    print("\n--- Per expected urgency ---", flush=True)
    for level in ("routine", "semi-urgent", "urgent"):
        subset = by_expected.get(level, [])
        if not subset:
            continue
        ok = sum(1 for r in subset if r["urgency_match"])
        print(f"  {level}: {ok}/{len(subset)} correct ({100 * ok / len(subset):.1f}%)", flush=True)

    if verbose:
        print("\n--- Per-sample details ---", flush=True)
        for i, r in enumerate(results):
            print(f"\n[{i + 1}] {r['transcript_preview']}", flush=True)
            print(f"    expected={r['expected_urgency']} pred={r['model_urgency']} match={r['urgency_match']}", flush=True)
            print(f"    correctness={r['correctness']:.3f} faithfulness={r['faithfulness']:.3f} relevance={r['relevance']:.3f}", flush=True)
            print(f"    summary: {r['model_summary'][:150]}...", flush=True)


async def main() -> None:
    # Ensure terminal output is visible immediately (no buffering)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Evaluate finetuned Ollama triage model on combined_triage_training.jsonl (correctness, faithfulness, relevance, urgency accuracy)."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "modelling" / "data" / "combined_triage_training.jsonl",
        help="Path to combined JSONL (transcript + output).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap total rows for quick runs.")
    parser.add_argument("--n-routine", type=int, default=50, help="Max routine samples.")
    parser.add_argument("--n-semi", type=int, default=30, help="Max semi-urgent samples.")
    parser.add_argument("--n-urgent", type=int, default=30, help="Max urgent samples.")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample details.")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: {args.data} not found", flush=True)
        sys.exit(1)

    gold = load_gold_set(args.data, args.n_routine, args.n_semi, args.n_urgent)
    if args.limit:
        gold = gold[: args.limit]

    print(f"Model: {settings.OLLAMA_MODEL_NAME} @ {settings.OLLAMA_BASE_URL}", flush=True)
    print(f"Loaded {len(gold)} gold examples from {args.data}", flush=True)
    print("", flush=True)

    results = []
    try:
        for i, g in enumerate(gold):
            print(f"[{i + 1}/{len(gold)}] {g['transcript'][:60]}... | expected={g['expected_urgency']}", end=" ", flush=True)
            try:
                r = await run_one(g)
                results.append(r)
                match_str = "OK" if r["urgency_match"] else "MISMATCH"
                print(f"-> pred={r['model_urgency']} {match_str} | cor={r['correctness']:.2f} faith={r['faithfulness']:.2f} rel={r['relevance']:.2f}", flush=True)
            except Exception as e:
                print(f"ERROR: {e}", flush=True)
                results.append({
                    "transcript_preview": g["transcript"][:80],
                    "expected_urgency": g["expected_urgency"],
                    "model_urgency": "error",
                    "model_summary": "",
                    "correctness": 0.0,
                    "faithfulness": 0.0,
                    "relevance": 0.0,
                    "urgency_match": False,
                })
    finally:
        # Always print final report (even on Ctrl+C or early exit)
        print("", flush=True)
        print_report(results, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
