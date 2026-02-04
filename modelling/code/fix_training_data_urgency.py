"""
Fix training_data.jsonl so outputs are ONLY urgency: "routine", "semi-urgent", or "urgent".
Also reassigns some rows by severity and adds synthetic ENT examples for balance.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IN_FILE = DATA_DIR / "training_data.jsonl"
OUT_FILE = DATA_DIR / "training_data.jsonl"

INSTRUCTION = "You are an ENT triage expert. Classify the urgency of this patient as routine, semi-urgent, or urgent based on their symptoms."


def parse_symptom_input(text: str) -> dict:
    """Extract fever value and severity counts from 'Patient presents with: ...' input."""
    fever_val = None
    severe_count = 0
    for part in text.replace("\n", " ").split():
        # e.g. (104.8) or (98.3)
        m = re.search(r"\((\d+\.?\d*)\)", part)
        if m and "fever" in text.lower().split(part)[0][-20:]:
            try:
                fever_val = float(m.group(1))
            except ValueError:
                pass
        if "severe" in part.lower():
            severe_count += 1
    # Also check for fever line explicitly
    if fever_val is None:
        fm = re.search(r"fever[:\s]+(?:\w+)\s*\((\d+\.?\d*)\)", text, re.I)
        if fm:
            try:
                fever_val = float(fm.group(1))
            except ValueError:
                pass
    return {"fever": fever_val, "severe_count": severe_count}


def infer_urgency_from_symptoms(text: str) -> str:
    """
    Infer urgency from symptom profile (fever + severity).
    Conservative: when in doubt, escalate. High fever + multiple severe -> semi-urgent or urgent.
    """
    info = parse_symptom_input(text)
    fever = info["fever"]
    severe = info["severe_count"]

    # Very high fever + many severe symptoms -> urgent
    if fever is not None and fever >= 104.0 and severe >= 4:
        return "urgent"
    if fever is not None and fever >= 103.0 and severe >= 3:
        return "semi-urgent"
    if fever is not None and fever >= 102.0 and severe >= 2:
        return "semi-urgent"
    # Default: routine
    return "routine"


def fix_existing_record(obj: dict) -> dict:
    """Fix one record: instruction + output = urgency only; optionally reassign urgency from input."""
    inp = obj.get("input", "")
    # Reassign urgency from symptom severity for numeric symptom profiles
    if "Patient presents with:" in inp and ("fever:" in inp or "Fever" in inp):
        urgency = infer_urgency_from_symptoms(inp)
    else:
        # Keep existing if present, else routine
        out = obj.get("output", "")
        if "URGENCY:" in out:
            u = out.split("URGENCY:")[-1].strip().split()[0].strip().lower()
            urgency = "semi-urgent" if u == "semi-urgent" else "urgent" if u == "urgent" else "routine"
        else:
            urgency = "routine"

    return {
        "instruction": INSTRUCTION,
        "input": inp,
        "output": urgency,
    }


def synthetic_ent_examples():
    """Generate synthetic ENT triage examples with clear urgent/semi-urgent/routine."""
    examples = []

    # --- ROUTINE ---
    routine_inputs = [
        "Patient has mild sore throat for 2 days. Feeling better with rest and tea. No fever.",
        "Mild nasal congestion for one week. No fever, no ear pain. Symptoms improving.",
        "Patient reports occasional mild ear discomfort, no discharge. No hearing change.",
        "Slight hoarseness for 3 days after a cold. No difficulty swallowing or breathing.",
        "Mild allergy symptoms: runny nose, sneezing. No fever or facial pain.",
        "Patient presents with:\nfever: mild (98.3)\nheadache: mild (1.4)\ncough: mild (2.7)\nfatigue: mild (2.1)\nbody_pain: mild (0.5)",
    ]
    for inp in routine_inputs:
        examples.append({"instruction": INSTRUCTION, "input": inp, "output": "routine"})

    # --- SEMI-URGENT ---
    semi_inputs = [
        "Sore throat worsening over 3 days with fever 101F. Difficulty swallowing solids. No breathing difficulty.",
        "Ear pain and discharge for 2 days. Mild fever. No dizziness or hearing loss.",
        "Moderate sore throat with fever 102F. Symptoms not improving with OTC meds. Patient has diabetes.",
        "Sinus pain and pressure for 5 days with fever 100.5F. Worsening headache.",
        "Patient presents with:\nfever: severe (103.2)\nheadache: severe (6.5)\ncough: moderate (5.0)\nfatigue: severe (6.0)\nbody_pain: severe (5.5)",
        "Worsening throat pain over 48 hours. Fever 101.5F. Able to swallow liquids. No stridor.",
        "Unilateral ear pain and hearing muffled for 2 days. No sudden hearing loss. Low-grade fever.",
    ]
    for inp in semi_inputs:
        examples.append({"instruction": INSTRUCTION, "input": inp, "output": "semi-urgent"})

    # --- URGENT ---
    urgent_inputs = [
        "Patient has severe sore throat, difficulty breathing, and stridor. Fever 102F. Needs immediate evaluation.",
        "Sudden complete hearing loss in one ear today. No trauma. No other neurological symptoms.",
        "Severe throat pain with inability to swallow saliva. Drooling. Suspected peritonsillar abscess.",
        "Immunocompromised patient (HIV) with severe sore throat and fever 103F. Symptoms worsening rapidly.",
        "Patient reports difficulty breathing, severe throat swelling, and wheezing. History of asthma.",
        "Sudden severe vertigo with vomiting. Cannot stand. Hearing unchanged. No other neurological signs.",
        "Child with high fever, severe throat pain, and difficulty breathing. Stridor noted by parent.",
        "Severe throat pain with fever 104F. Difficulty swallowing. Immunocompromised. Spreading neck swelling.",
    ]
    for inp in urgent_inputs:
        examples.append({"instruction": INSTRUCTION, "input": inp, "output": "urgent"})

    return examples


def main():
    fixed = []
    with open(IN_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fixed.append(fix_existing_record(obj))

    synthetic = synthetic_ent_examples()
    all_records = fixed + synthetic

    with open(OUT_FILE, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    counts = {}
    for rec in all_records:
        u = rec["output"]
        counts[u] = counts.get(u, 0) + 1

    print(f"Wrote {len(all_records)} examples to {OUT_FILE}")
    print(f"  From file: {len(fixed)} (fixed)")
    print(f"  Synthetic: {len(synthetic)}")
    print(f"  Urgency distribution: {counts}")


if __name__ == "__main__":
    main()
