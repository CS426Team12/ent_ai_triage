"""
Run summary validation over an evaluation dataset.
Computes correctness, faithfulness, and relevance for each (transcript, summary) pair.
Use --summary-source=reference to score reference summaries; use --summary-source=api to call triage API.
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root or from validation/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.metrics import validate_summary, ValidationScores

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_EVAL_FILE = DATA_DIR / "synthetic_eval.jsonl"


def load_eval_data(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_validation(
    eval_path: Path,
    summary_source: str = "reference",
    api_url: str | None = None,
) -> list[tuple[dict, ValidationScores]]:
    """
    summary_source: "reference" (use reference_summary from file) or "api" (call triage API for summary).
    """
    rows = load_eval_data(eval_path)
    results = []

    if summary_source == "api" and api_url:
        try:
            import httpx
        except ImportError:
            raise RuntimeError("Install httpx to use --summary-source=api")

    for row in rows:
        transcript = row.get("transcript", "")
        if summary_source == "reference":
            summary = row.get("reference_summary", "")
        elif summary_source == "api" and api_url:
            try:
                with httpx.Client(timeout=30.0) as client:
                    r = client.post(
                        f"{api_url.rstrip('/')}/ai/triage",
                        json={"patient_id": "eval", "transcript": transcript},
                    )
                    r.raise_for_status()
                    data = r.json()
                    summary = data.get("summary", "")
            except Exception as e:
                summary = f"[API error: {e}]"
        else:
            summary = row.get("reference_summary", "")

        scores = validate_summary(transcript, summary)
        results.append((row, scores))

    return results


def print_report(results: list[tuple[dict, ValidationScores]], verbose: bool = False):
    if not results:
        print("No results.")
        return

    n = len(results)
    avg_c = sum(r[1].correctness for r in results) / n
    avg_f = sum(r[1].faithfulness for r in results) / n
    avg_r = sum(r[1].relevance for r in results) / n

    print("=" * 60)
    print("SUMMARY VALIDATION REPORT")
    print("=" * 60)
    print(f"  Correctness  (avg): {avg_c:.3f}")
    print(f"  Faithfulness (avg): {avg_f:.3f}")
    print(f"  Relevance    (avg): {avg_r:.3f}")
    print(f"  N samples: {n}")
    print("=" * 60)

    if verbose:
        for i, (row, scores) in enumerate(results):
            print(f"\n--- Sample {i+1} ---")
            print(f"  Transcript (excerpt): {row.get('transcript', '')[:120]}...")
            print(f"  Summary (excerpt):   {row.get('reference_summary', '')[:120]}...")
            print(f"  Correctness: {scores.correctness}  Faithfulness: {scores.faithfulness}  Relevance: {scores.relevance}")


def main():
    parser = argparse.ArgumentParser(description="Validate LLM triage summaries (correctness, faithfulness, relevance).")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="JSONL eval file (transcript, reference_summary, urgency).")
    parser.add_argument("--summary-source", choices=("reference", "api"), default="reference", help="Use reference_summary from file or fetch summary from triage API.")
    parser.add_argument("--api-url", type=str, default=None, help="Triage API base URL (e.g. http://localhost:8100) when --summary-source=api.")
    parser.add_argument("--verbose", action="store_true", help="Print per-sample scores.")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic eval data to --eval-file then run validation.")
    args = parser.parse_args()

    if args.generate:
        from validation.synthetic_data import generate_synthetic_eval
        generate_synthetic_eval(out_path=args.eval_file)
        print(f"Generated eval data at {args.eval_file}")

    if not args.eval_file.exists():
        print(f"Eval file not found: {args.eval_file}. Run with --generate first.")
        sys.exit(1)

    if args.summary_source == "api" and not args.api_url:
        print("--api-url required when --summary-source=api")
        sys.exit(1)

    results = run_validation(
        eval_path=args.eval_file,
        summary_source=args.summary_source,
        api_url=args.api_url,
    )
    print_report(results, verbose=args.verbose)


if __name__ == "__main__":
    main()
