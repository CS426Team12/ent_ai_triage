"""
Prepare finetuning data: filter 3+ sentence summaries and optionally generate more.

Usage (from ent_ai_triage/ or repo root):
  python modelling/code/prepare_training_data.py
  python modelling/code/prepare_training_data.py --no-extra
  python modelling/code/prepare_training_data.py --extra-count 2000 --extra-semi 500
  python modelling/code/prepare_training_data.py --extra-count 500 --no-combine
"""

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "data"


def main():
    parser = argparse.ArgumentParser(description="Prepare triage training data (3+ sentence summaries)")
    parser.add_argument("--no-extra", action="store_true", help="Skip generating extra synthetic data")
    parser.add_argument("--extra-count", type=int, default=1500, help="Number of extra examples to generate")
    parser.add_argument("--extra-semi", type=int, default=0, help="Append N semi-urgent-only examples before combining")
    parser.add_argument("--semi-ratio", type=float, default=0.4, help="Fraction of semi-urgent in extra generation (default 0.4)")
    parser.add_argument("--no-combine", action="store_true", help="Do not create combined_triage_training.jsonl")
    args = parser.parse_args()

    # 1. Filter existing data to 3+ sentence summaries
    ensure_script = CODE_DIR / "ensure_summary_length.py"
    print("Running ensure_summary_length.py...")
    r = subprocess.run([sys.executable, str(ensure_script)], cwd=str(CODE_DIR.parent.parent))
    if r.returncode != 0:
        sys.exit(r.returncode)

    if not args.no_extra:
        # 2. Generate extra synthetic data (40% semi-urgent by default)
        gen_script = CODE_DIR / "generate_more_triage_data.py"
        extra_out = str(DATA_DIR / "triage_training_data_extra.jsonl")
        print("\nRunning generate_more_triage_data.py...")
        r = subprocess.run(
            [
                sys.executable, str(gen_script),
                "--count", str(args.extra_count),
                "--semi-ratio", str(args.semi_ratio),
                "--out", extra_out,
            ],
            cwd=str(CODE_DIR.parent.parent),
        )
        if r.returncode != 0:
            sys.exit(r.returncode)
        if args.extra_semi > 0:
            print(f"\nAppending {args.extra_semi} semi-urgent-only examples...")
            r = subprocess.run(
                [
                    sys.executable, str(gen_script),
                    "--semi-only", str(args.extra_semi),
                    "--append",
                    "--out", extra_out,
                ],
                cwd=str(CODE_DIR.parent.parent),
            )
            if r.returncode != 0:
                sys.exit(r.returncode)

    if not args.no_combine:
        # 3. Combine into one file
        min3 = DATA_DIR / "triage_training_data_min3sentences.jsonl"
        extra = DATA_DIR / "triage_training_data_extra.jsonl"
        combined = DATA_DIR / "combined_triage_training.jsonl"
        if min3.exists():
            with open(combined, "w") as out:
                for f in [min3, extra] if extra.exists() else [min3]:
                    with open(f) as inp:
                        out.write(inp.read())
            n = sum(1 for _ in open(combined))
            print(f"\nCreated {combined.name} with {n} examples.")
        else:
            print("\nSkipping combine: triage_training_data_min3sentences.jsonl not found.")

    print("\nDone. Use combined_triage_training.jsonl in the finetuning notebook.")


if __name__ == "__main__":
    main()
