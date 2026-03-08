"""
Ensure summaries in triage_training_data.jsonl have at least 3 sentences.
Use before finetuning to filter or report short summaries.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_FILE = DATA_DIR / "triage_training_data.jsonl"
OUT_FILE = DATA_DIR / "triage_training_data_min3sentences.jsonl"


def count_sentences(text: str) -> int:
    """Count sentences (split on . ! ?)."""
    if not text or not text.strip():
        return 0
    # Split on sentence-ending punctuation
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])


def extract_summary(output: str) -> str:
    """Extract SUMMARY section from output."""
    m = re.search(r"SUMMARY:\s*(.+?)(?=FINDINGS:|FLAGS:|$)", output, re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "").strip()


def main():
    in_path = IN_FILE
    if not in_path.exists():
        print(f"File not found: {in_path}")
        return

    data = []
    with open(in_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    kept = []
    short = []
    for i, rec in enumerate(data):
        output = rec.get("output", "")
        summary = extract_summary(output)
        n = count_sentences(summary)
        if n >= 3:
            kept.append(rec)
        else:
            short.append((i, n, summary[:80]))

    print(f"Total: {len(data)}")
    print(f"Kept (3+ sentences): {len(kept)}")
    print(f"Filtered (short): {len(short)}")
    if short:
        print("\nSample short summaries:")
        for idx, n, s in short[:5]:
            print(f"  [{idx}] {n} sent: {s}...")

    out_path = OUT_FILE
    with open(out_path, "w") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    print(f"\nWrote filtered data to {out_path}")
    print("Use this file for finetuning: triage_training_data_min3sentences.jsonl")


if __name__ == "__main__":
    main()
