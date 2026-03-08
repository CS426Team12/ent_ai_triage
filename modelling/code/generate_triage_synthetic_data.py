"""
Generate synthetic ENT triage data for LLM finetuning.
Transcript format: Lex/Twilio slot:value. (e.g. chief_complaint:I'm coughing. symptom_duration:1 month)
Output format: SUMMARY, FINDINGS, FLAGS, URGENCY, REASONING (full structured response)
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_FILE = DATA_DIR / "triage_training_data.jsonl"

# Lex slot names (Twilio/Lex format)
SLOTS = [
    "chief_complaint",
    "symptom_duration",
    "symptom_severity",
    "symptom_progression",
    "aggravating_factors",
    "relieving_factors",
    "associated_symptoms",
    "red_flags",
    "risk_factors",
]

# Synthetic slot values by urgency
ROUTINE_SLOTS = {
    "chief_complaint": [
        "mild sore throat",
        "congestion",
        "runny nose",
        "mild ear discomfort",
        "hoarse voice",
        "post-nasal drip",
        "mild cough",
        "sinus pressure",
    ],
    "symptom_duration": ["1 day", "2 days", "3 days", "1 week", "5 days"],
    "symptom_severity": ["mild", "slight", "annoying", "2 out of 10", "3 out of 10"],
    "symptom_progression": ["improving", "stable", "getting better", "same"],
    "aggravating_factors": ["swallowing", "talking", "cold water", "None"],
    "relieving_factors": ["warm tea", "rest", "lozenges", "ibuprofen"],
    "associated_symptoms": ["No", "mild fatigue", "slight headache"],
    "red_flags": ["No"],
    "risk_factors": ["No"],
}

SEMI_URGENT_SLOTS = {
    "chief_complaint": [
        "severe sore throat",
        "ear pain with discharge",
        "worsening sinus pain",
        "persistent throat pain",
        "moderate earache",
        "throat pain with fever",
        "sinus pain with fever",
        "persistent cough with fatigue",
        "worsening hoarseness",
        "nasal congestion with facial pressure",
    ],
    "symptom_duration": ["2 days", "3 days", "4 days", "5 days", "6 days", "1 week"],
    "symptom_severity": ["moderate", "4 out of 10", "5 out of 10", "6 out of 10", "7 out of 10", "significant"],
    "symptom_progression": ["worsening", "getting worse", "not improving", "fluctuating"],
    "aggravating_factors": ["swallowing", "eating", "talking"],
    "relieving_factors": ["ibuprofen helps a little", "rest", "warm liquids"],
    "associated_symptoms": ["fever", "headache", "fatigue"],
    "red_flags": ["No", "fever present"],
    "risk_factors": ["diabetes", "No"],
}

URGENT_SLOTS = {
    "chief_complaint": [
        "difficulty breathing",
        "severe throat pain",
        "cannot swallow",
        "stridor",
        "severe dizziness",
        "sudden hearing loss",
        "facial swelling",
    ],
    "symptom_duration": ["few hours", "1 day", "2 days"],
    "symptom_severity": ["severe", "9 out of 10", "unbearable"],
    "symptom_progression": ["rapidly worsening", "getting worse quickly"],
    "aggravating_factors": ["swallowing impossible", "any movement"],
    "relieving_factors": ["Nothing helps"],
    "associated_symptoms": ["fever", "chills", "drooling"],
    "red_flags": ["Yes", "difficulty breathing", "cannot swallow"],
    "risk_factors": ["immunocompromised", "diabetes", "HIV"],
}


def slots_to_transcript(slots: dict) -> str:
    """Convert slot dict to Lex format: slot:value. slot:value."""
    parts = [f"{k}:{v}" for k, v in slots.items() if v]
    return ". ".join(parts)


def build_slots(template: dict, rng: random.Random) -> dict:
    out = {}
    for k, vals in template.items():
        out[k] = rng.choice(vals) if isinstance(vals, list) else vals
    return out


def make_output(summary: str, findings: list, flags: list, urgency: str, reasoning: str) -> str:
    flags_str = ", ".join(f"[{f['tag']}] {f['keyword']}" for f in flags)
    findings_str = "\n".join(f"- {f}" for f in findings)
    return f"""SUMMARY: {summary}
FINDINGS:
{findings_str}
FLAGS: {flags_str}
URGENCY: {urgency}
REASONING: {reasoning}"""


# Pre-built (transcript_slots_str, output) tuples for variety
ROUTINE_EXAMPLES = [
    (
        {"chief_complaint": "mild sore throat", "symptom_duration": "3 days", "symptom_severity": "mild",
         "symptom_progression": "improving", "aggravating_factors": "swallowing", "relieving_factors": "warm tea",
         "associated_symptoms": "No", "red_flags": "No", "risk_factors": "No"},
        make_output(
            "Patient presents with mild sore throat of 3 days duration. Symptoms are improving with warm liquids. No red flags.",
            ["Mild sore throat", "3-day duration", "Improving trend"],
            [{"tag": "SYMPTOM", "keyword": "sore throat"}, {"tag": "SEVERITY", "keyword": "mild"},
             {"tag": "DURATION", "keyword": "3 days"}, {"tag": "PROGRESSION", "keyword": "improving"},
             {"tag": "RELIEVING_FACTORS", "keyword": "warm tea"}],
            "routine",
            "Mild symptoms, improving trend, no red flags. Safe to schedule routine appointment.",
        ),
    ),
    (
        {"chief_complaint": "I'm coughing", "symptom_duration": "1 month", "symptom_severity": "moderate",
         "symptom_progression": "better", "aggravating_factors": "Drinking alcohol",
         "relieving_factors": "Physical activity", "associated_symptoms": "No", "red_flags": "No", "risk_factors": "No"},
        make_output(
            "Patient with chronic cough of 1 month, moderate severity, now improving. Alcohol worsens; physical activity helps. No red flags.",
            ["Chronic cough", "1-month duration", "Moderate severity", "Improving"],
            [{"tag": "SYMPTOM", "keyword": "cough"}, {"tag": "DURATION", "keyword": "1 month"},
             {"tag": "SEVERITY", "keyword": "moderate"}, {"tag": "PROGRESSION", "keyword": "improving"},
             {"tag": "AGGRAVATING_FACTORS", "keyword": "alcohol"}, {"tag": "RELIEVING_FACTORS", "keyword": "physical activity"}],
            "routine",
            "Chronic cough improving, no red flags. Routine evaluation appropriate.",
        ),
    ),
    (
        {"chief_complaint": "congestion", "symptom_duration": "5 days", "symptom_severity": "mild",
         "symptom_progression": "stable", "aggravating_factors": "None", "relieving_factors": "decongestant",
         "associated_symptoms": "mild fatigue", "red_flags": "No", "risk_factors": "No"},
        make_output(
            "Patient with mild congestion for 5 days, stable. Responds to decongestant. No fever or red flags.",
            ["Nasal congestion", "5-day duration", "Stable", "Mild fatigue"],
            [{"tag": "SYMPTOM", "keyword": "congestion"}, {"tag": "SEVERITY", "keyword": "mild"},
             {"tag": "DURATION", "keyword": "5 days"}, {"tag": "ASSOCIATED_SYMPTOMS", "keyword": "fatigue"}],
            "routine",
            "Mild stable congestion. Routine appointment sufficient.",
        ),
    ),
]

SEMI_URGENT_EXAMPLES = [
    (
        {"chief_complaint": "severe sore throat", "symptom_duration": "4 days", "symptom_severity": "moderate",
         "symptom_progression": "worsening", "aggravating_factors": "swallowing",
         "relieving_factors": "ibuprofen helps a little", "associated_symptoms": "fever",
         "red_flags": "No", "risk_factors": "No"},
        make_output(
            "Patient with severe sore throat, fever, worsening over 4 days. Pain on swallowing. Ibuprofen provides partial relief.",
            ["Severe sore throat", "Fever", "Worsening", "Pain on swallowing"],
            [{"tag": "SYMPTOM", "keyword": "sore throat"}, {"tag": "SEVERITY", "keyword": "severe"},
             {"tag": "PROGRESSION", "keyword": "worsening"}, {"tag": "ASSOCIATED_SYMPTOMS", "keyword": "fever"}],
            "semi-urgent",
            "Fever plus worsening throat pain warrants evaluation within 24-48 hours.",
        ),
    ),
]

URGENT_EXAMPLES = [
    (
        {"chief_complaint": "difficulty breathing", "symptom_duration": "few hours",
         "symptom_severity": "severe", "symptom_progression": "rapidly worsening",
         "aggravating_factors": "any movement", "relieving_factors": "Nothing helps",
         "associated_symptoms": "fever", "red_flags": "Yes", "risk_factors": "No"},
        make_output(
            "Patient with acute difficulty breathing, rapidly worsening. Severe presentation. Critical red flag.",
            ["Difficulty breathing", "Rapid deterioration", "Severe presentation"],
            [{"tag": "RED_FLAG", "keyword": "difficulty breathing"}, {"tag": "SEVERITY", "keyword": "severe"},
             {"tag": "PROGRESSION", "keyword": "rapidly worsening"}],
            "urgent",
            "Difficulty breathing is a critical red flag. Same-day evaluation required.",
        ),
    ),
    (
        {"chief_complaint": "cannot swallow", "symptom_duration": "1 day", "symptom_severity": "unbearable",
         "symptom_progression": "getting worse quickly", "aggravating_factors": "swallowing impossible",
         "relieving_factors": "Nothing helps", "associated_symptoms": "drooling", "red_flags": "Yes",
         "risk_factors": "No"},
        make_output(
            "Patient unable to swallow, drooling. Unbearable pain. Critical airway concern.",
            ["Cannot swallow", "Drooling", "Unbearable pain"],
            [{"tag": "RED_FLAG", "keyword": "cannot swallow"}, {"tag": "SEVERITY", "keyword": "unbearable"}],
            "urgent",
            "Inability to swallow with drooling suggests potential airway compromise. Urgent same-day evaluation.",
        ),
    ),
]


def generate_more(rng: random.Random, n_routine: int, n_semi: int, n_urgent: int) -> list:
    examples = []
    for _ in range(n_routine):
        slots = build_slots(ROUTINE_SLOTS, rng)
        transcript = slots_to_transcript(slots)
        urgency = "routine"
        flags = [
            {"tag": "SYMPTOM", "keyword": slots.get("chief_complaint", "symptoms")[:30]},
            {"tag": "SEVERITY", "keyword": slots.get("symptom_severity", "mild")},
            {"tag": "DURATION", "keyword": slots.get("symptom_duration", "few days")},
        ]
        findings = [slots.get("chief_complaint", "ENT symptoms"), f"{slots.get('symptom_duration', '')} duration"]
        summary = f"Patient with {slots.get('chief_complaint', 'symptoms')} for {slots.get('symptom_duration', '')}. Symptoms are {slots.get('symptom_progression', 'stable')}. No red flags or concerning symptoms."
        reasoning = "Mild symptoms, no red flags. Routine evaluation."
        examples.append((transcript, make_output(summary, findings, flags, urgency, reasoning)))
    for _ in range(n_semi):
        slots = build_slots(SEMI_URGENT_SLOTS, rng)
        transcript = slots_to_transcript(slots)
        flags = [{"tag": "SEVERITY", "keyword": "moderate"}, {"tag": "PROGRESSION", "keyword": "worsening"}]
        findings = [slots.get("chief_complaint", "symptoms"), "Worsening", "May have fever"]
        summary = f"Patient presents with {slots.get('chief_complaint', 'symptoms')} that is worsening. Symptoms warrant evaluation within 24-48 hours. No critical red flags but moderate severity."
        examples.append((transcript, make_output(summary, findings, flags, "semi-urgent",
                                                "Moderate worsening symptoms warrant evaluation within 24-48 hours.")))
    for _ in range(n_urgent):
        slots = build_slots(URGENT_SLOTS, rng)
        transcript = slots_to_transcript(slots)
        flags = [{"tag": "RED_FLAG", "keyword": slots.get("chief_complaint", "critical")[:40]}]
        findings = [slots.get("chief_complaint", "Critical symptoms"), "Urgent evaluation needed"]
        summary = f"Patient presents with {slots.get('chief_complaint', 'critical symptoms')}. Critical red flag detected. Same-day urgent evaluation required."
        examples.append((transcript, make_output(summary, findings, flags, "urgent",
                                                "Critical red flag. Same-day evaluation required.")))
    return examples


def main():
    rng = random.Random(42)
    data = []

    # Add curated examples (repeated for weight)
    for _ in range(80):
        for slots, output in ROUTINE_EXAMPLES:
            data.append({"transcript": slots_to_transcript(slots), "output": output})
    for _ in range(60):
        for slots, output in SEMI_URGENT_EXAMPLES:
            data.append({"transcript": slots_to_transcript(slots), "output": output})
    for _ in range(60):
        for slots, output in URGENT_EXAMPLES:
            data.append({"transcript": slots_to_transcript(slots), "output": output})

    # Add generated variety
    more = generate_more(rng, n_routine=200, n_semi=100, n_urgent=80)
    for transcript, output in more:
        data.append({"transcript": transcript, "output": output})

    rng.shuffle(data)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        for rec in data:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(data)} examples to {OUT_FILE}")


if __name__ == "__main__":
    main()
