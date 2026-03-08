#!/usr/bin/env python3
"""
Evaluate urgency classification across the triage pipeline.

Builds a gold set from combined_triage_training.jsonl, runs each through
call_ollama + validate_urgency_classification, and reports accuracy per stage.

Run from ent_ai_triage directory:
  python scripts/eval_urgency.py
  python scripts/eval_urgency.py --limit 20   # Quick run

Requires: Ollama running with ent-triage-qwen model.
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add parent so we can import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Parse --trace early so DEBUG_URGENCY is set before importing app
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", action="store_true")
    p.add_argument("--data", default="modelling/data/combined_triage_training.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n-routine", type=int, default=50)
    p.add_argument("--n-semi", type=int, default=30)
    p.add_argument("--n-urgent", type=int, default=30)
    return p.parse_args()

_early_args = _parse_args()
if _early_args.trace:
    os.environ["DEBUG_URGENCY"] = "1"
    if Path("urgency_trace.jsonl").exists():
        Path("urgency_trace.jsonl").unlink()

from app.ollama_client import call_ollama, validate_urgency_classification
from app.ml_client import predict_urgency


def parse_urgency_from_output(output: str) -> str | None:
    """Extract URGENCY from training output. Returns routine|semi-urgent|urgent or None."""
    m = re.search(r'URGENCY:\s*(\w+(?:-\w+)?)', output, re.IGNORECASE)
    if not m:
        return None
    u = m.group(1).lower()
    return u if u in ("routine", "semi-urgent", "urgent") else None


def load_gold_set(data_path: Path, n_routine: int = 50, n_semi: int = 30, n_urgent: int = 30) -> list[dict]:
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
    """Run one example through the pipeline."""
    transcript = gold["transcript"]
    expected = gold["expected_urgency"]
    patient_history = {}

    llm_result = await call_ollama(transcript, patient_history)
    parsed_urgency = llm_result.get("urgency", "routine")
    summary = llm_result.get("summary", "")
    flags_data = llm_result.get("flags", [])

    ml_prediction = predict_urgency(transcript)
    ml_confidence = ml_prediction.get("confidence", 0.0)

    final_urgency, _ = validate_urgency_classification(
        transcript=transcript,
        llm_urgency=parsed_urgency,
        flags=flags_data,
        patient_history=patient_history,
        ml_confidence=ml_confidence,
        summary=summary,
    )

    return {
        "transcript_preview": transcript[:80] + "..." if len(transcript) > 80 else transcript,
        "expected_urgency": expected,
        "parsed_urgency": parsed_urgency,
        "final_urgency": final_urgency,
        "parsed_correct": parsed_urgency == expected,
        "final_correct": final_urgency == expected,
    }


def print_metrics(results: list[dict]) -> None:
    """Print accuracy and confusion matrix."""
    total = len(results)
    parsed_correct = sum(1 for r in results if r["parsed_correct"])
    final_correct = sum(1 for r in results if r["final_correct"])

    print("\n" + "=" * 60)
    print("URGENCY EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Parsed (LLM) accuracy:  {parsed_correct}/{total} = {100*parsed_correct/total:.1f}%")
    print(f"Final (after validation) accuracy: {final_correct}/{total} = {100*final_correct/total:.1f}%")

    # Per-urgency breakdown
    by_expected = defaultdict(list)
    for r in results:
        by_expected[r["expected_urgency"]].append(r)

    print("\n--- Per expected urgency ---")
    for level in ("routine", "semi-urgent", "urgent"):
        subset = by_expected.get(level, [])
        if not subset:
            continue
        parsed_ok = sum(1 for r in subset if r["parsed_correct"])
        final_ok = sum(1 for r in subset if r["final_correct"])
        print(f"  {level}: {final_ok}/{len(subset)} final correct, {parsed_ok}/{len(subset)} parsed correct")

    # Confusion: expected vs final
    print("\n--- Confusion matrix (expected -> predicted) ---")
    levels = ["routine", "semi-urgent", "urgent"]
    matrix = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r["expected_urgency"]][r["final_urgency"]] += 1
    header = "              " + "".join(f"{l:>14}" for l in levels)
    print(header)
    for exp in levels:
        row = f"{exp:14}" + "".join(f"{matrix[exp].get(pred,0):>14}" for pred in levels)
        print(row)

    # List misclassified
    misclassified = [r for r in results if not r["final_correct"]]
    if misclassified:
        print(f"\n--- Misclassified ({len(misclassified)}) ---")
        for r in misclassified[:15]:
            print(f"  expected={r['expected_urgency']} -> final={r['final_urgency']} (parsed={r['parsed_urgency']}) | {r['transcript_preview']}")


async def main():
    args = _early_args
    root = Path(__file__).resolve().parent.parent
    data_path = root / args.data
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        sys.exit(1)

    gold = load_gold_set(data_path, args.n_routine, args.n_semi, args.n_urgent)
    if args.limit:
        gold = gold[: args.limit]
    print(f"Loaded {len(gold)} gold examples (routine/semi/urgent)")

    results = []
    for i, g in enumerate(gold):
        print(f"  [{i+1}/{len(gold)}] {g['expected_urgency']}...", end=" ", flush=True)
        try:
            r = await run_one(g)
            results.append(r)
            status = "OK" if r["final_correct"] else "MISMATCH"
            print(status)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "transcript_preview": g["transcript"][:80],
                "expected_urgency": g["expected_urgency"],
                "parsed_urgency": "error",
                "final_urgency": "error",
                "parsed_correct": False,
                "final_correct": False,
            })

    print_metrics(results)


if __name__ == "__main__":
    asyncio.run(main())
