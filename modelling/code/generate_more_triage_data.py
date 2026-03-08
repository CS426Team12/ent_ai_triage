"""
Generate additional synthetic ENT triage data for finetuning.
All summaries are at least 3 sentences.

Usage:
  python modelling/code/generate_more_triage_data.py
  python modelling/code/generate_more_triage_data.py --count 2000 --out triage_extra.jsonl
"""

import json
import random
import argparse
import sys
from pathlib import Path

# Allow import when run from repo root
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from generate_triage_synthetic_data import (
    slots_to_transcript,
    build_slots,
    make_output,
    ROUTINE_SLOTS,
    SEMI_URGENT_SLOTS,
    URGENT_SLOTS,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUT = DATA_DIR / "triage_training_data_extra.jsonl"

# 3-sentence summary templates (sentence1. sentence2. sentence3.)
ROUTINE_SUMMARY_TEMPLATES = [
    "Patient presents with {complaint} for {duration}. Symptoms are {progression} and severity is {severity}. No red flags; routine follow-up is appropriate.",
    "Caller reports {complaint} lasting {duration} with {severity} severity. Progression is {progression}. No concerning red flags or risk factors.",
    "Patient describes {complaint}, {duration} in duration, currently {progression}. Severity is {severity}. Safe for routine scheduling.",
]

SEMI_URGENT_SUMMARY_TEMPLATES = [
    "Patient has {complaint} that has been {progression} over {duration}. Severity is {severity}. Evaluation within 24-48 hours is recommended.",
    "Caller reports {complaint} with {severity} severity, worsening over {duration}. Associated symptoms may be present. Same-week evaluation warranted.",
    "Patient presents with {complaint} lasting {duration}; symptoms are {progression}. Moderate severity. Should be seen within 24-48 hours.",
    "Patient reports {complaint} for {duration}. Symptoms are {progression} and {severity}. Semi-urgent: evaluation within 24-48 hours recommended.",
    "Caller describes {complaint}, {duration} in duration. Severity {severity}, progression {progression}. Needs evaluation within 24-48 hours; not same-day urgent.",
    "Patient with {complaint} for {duration}. {severity} severity, {progression}. Semi-urgent triage: schedule within 24-48 hours.",
]

URGENT_SUMMARY_TEMPLATES = [
    "Patient reports {complaint} with rapid onset; symptoms are {progression}. This represents a critical red flag. Same-day urgent evaluation is required.",
    "Caller describes {complaint} of {duration} with severe presentation. Critical red flag identified. Immediate same-day evaluation needed.",
    "Patient with {complaint}; presentation is severe and {progression}. Red flag present. Urgent same-day assessment required.",
]


def make_summary_3sent(template: str, slots: dict) -> str:
    """Fill template with slot values; ensure 3 sentences."""
    complaint = slots.get("chief_complaint", "ENT symptoms")
    duration = slots.get("symptom_duration", "several days")
    severity = slots.get("symptom_severity", "moderate")
    progression = slots.get("symptom_progression", "stable")
    return template.format(
        complaint=complaint,
        duration=duration,
        severity=severity,
        progression=progression,
    )


def generate_routine(rng: random.Random, n: int) -> list:
    out = []
    for _ in range(n):
        slots = build_slots(ROUTINE_SLOTS, rng)
        transcript = slots_to_transcript(slots)
        template = rng.choice(ROUTINE_SUMMARY_TEMPLATES)
        summary = make_summary_3sent(template, slots)
        flags = [
            {"tag": "SYMPTOM", "keyword": (slots.get("chief_complaint") or "symptoms")[:35]},
            {"tag": "SEVERITY", "keyword": slots.get("symptom_severity", "mild")},
            {"tag": "DURATION", "keyword": slots.get("symptom_duration", "few days")},
            {"tag": "PROGRESSION", "keyword": slots.get("symptom_progression", "stable")},
        ]
        if slots.get("relieving_factors") and slots.get("relieving_factors") != "None":
            flags.append({"tag": "RELIEVING_FACTORS", "keyword": slots["relieving_factors"][:30]})
        findings = [
            slots.get("chief_complaint", "ENT symptoms"),
            f"{slots.get('symptom_duration', '')} duration",
            f"{slots.get('symptom_progression', 'stable')}",
        ]
        reasoning = "Mild or stable symptoms with no red flags. Routine appointment is appropriate."
        output = make_output(summary, findings, flags, "routine", reasoning)
        out.append({"transcript": transcript, "output": output})
    return out


def generate_semi_urgent(rng: random.Random, n: int) -> list:
    out = []
    for _ in range(n):
        slots = build_slots(SEMI_URGENT_SLOTS, rng)
        transcript = slots_to_transcript(slots)
        template = rng.choice(SEMI_URGENT_SUMMARY_TEMPLATES)
        summary = make_summary_3sent(template, slots)
        flags = [
            {"tag": "SYMPTOM", "keyword": (slots.get("chief_complaint") or "symptoms")[:35]},
            {"tag": "SEVERITY", "keyword": slots.get("symptom_severity", "moderate")},
            {"tag": "PROGRESSION", "keyword": slots.get("symptom_progression", "worsening")},
        ]
        findings = [
            slots.get("chief_complaint", "symptoms"),
            "Worsening or not improving",
            "Evaluation within 24-48 hours recommended",
        ]
        reasoning = "Moderate severity with worsening symptoms. Should be seen within 24-48 hours."
        output = make_output(summary, findings, flags, "semi-urgent", reasoning)
        out.append({"transcript": transcript, "output": output})
    return out


def generate_urgent(rng: random.Random, n: int) -> list:
    out = []
    for _ in range(n):
        slots = build_slots(URGENT_SLOTS, rng)
        transcript = slots_to_transcript(slots)
        template = rng.choice(URGENT_SUMMARY_TEMPLATES)
        summary = make_summary_3sent(template, slots)
        complaint = (slots.get("chief_complaint") or "critical symptoms")[:40]
        flags = [
            {"tag": "RED_FLAG", "keyword": complaint},
            {"tag": "SEVERITY", "keyword": slots.get("symptom_severity", "severe")},
            {"tag": "PROGRESSION", "keyword": slots.get("symptom_progression", "rapidly worsening")},
        ]
        findings = [
            complaint,
            "Urgent evaluation needed",
            "Same-day assessment required",
        ]
        reasoning = "Critical red flag. Same-day urgent evaluation required."
        output = make_output(summary, findings, flags, "urgent", reasoning)
        out.append({"transcript": transcript, "output": output})
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate extra triage training data (3+ sentence summaries)")
    parser.add_argument("--count", type=int, default=1500, help="Total examples to generate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output jsonl path")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--semi-ratio", type=float, default=0.4, help="Fraction of semi-urgent (default 0.4)")
    parser.add_argument("--semi-only", type=int, default=None, help="Generate N semi-urgent-only examples (overrides --count)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.semi_only is not None:
        data = generate_semi_urgent(rng, args.semi_only)
    else:
        total = args.count
        # Default mix: 40% routine, 40% semi-urgent, 20% urgent
        semi_ratio = max(0.0, min(1.0, args.semi_ratio))
        n_semi = int(total * semi_ratio)
        n_urgent = int(total * 0.2)
        n_routine = total - n_semi - n_urgent
        if n_routine < 0:
            n_routine = 0
            n_semi = total - n_urgent

        data = []
        data.extend(generate_routine(rng, n_routine))
        data.extend(generate_semi_urgent(rng, n_semi))
        data.extend(generate_urgent(rng, n_urgent))
    if args.semi_only is None:
        rng.shuffle(data)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(out_path, mode) as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")

    if args.semi_only is not None:
        print(f"Wrote {len(data)} semi-urgent examples to {out_path}")
    else:
        print(f"Wrote {len(data)} examples to {out_path}")
        print(f"  routine: {n_routine}, semi-urgent: {n_semi}, urgent: {n_urgent}")
    print("All summaries are at least 3 sentences.")


if __name__ == "__main__":
    main()
