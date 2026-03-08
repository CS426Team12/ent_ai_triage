"""
Add context/instruction to training data so each sample has full input (instruction + transcript)
instead of raw transcript only.

Usage (from ent_ai_triage/ or repo root):
  python modelling/code/add_context_to_training_data.py
  python modelling/code/add_context_to_training_data.py --in combined_triage_training.jsonl
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_IN = DATA_DIR / "combined_triage_training.jsonl"

INSTRUCTION = """You are an ENT triage expert. Generate a 3+ sentence clinical summary, classify the urgency (routine, semi-urgent, or urgent), identify flags, and provide reasoning for this patient based on the call transcript below.

TRANSCRIPT:
"""


def add_context(record: dict) -> dict:
    transcript = record.get("transcript", "")
    if not transcript and "input" in record:
        return record
    record["input"] = INSTRUCTION + transcript
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default=str(DEFAULT_IN), help="Input jsonl path")
    parser.add_argument("--out", dest="outfile", default=None, help="Output path (default: overwrite input)")
    args = parser.parse_args()

    in_path = Path(args.infile)
    out_path = Path(args.outfile) if args.outfile else in_path

    if not in_path.exists():
        print(f"File not found: {in_path}")
        return

    data = []
    with open(in_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    transformed = [add_context(r) for r in data]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in transformed:
            f.write(json.dumps(r) + "\n")

    print(f"Added context to {len(transformed)} examples. Wrote to {out_path}")


if __name__ == "__main__":
    main()
