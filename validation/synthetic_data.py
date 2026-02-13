"""
Generate synthetic evaluation data for summary validation.
Produces (transcript, reference_summary, urgency) for correctness/faithfulness/relevance metrics.
"""

import json
import random
from pathlib import Path

# Default output under validation/data
DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_FILE = DATA_DIR / "synthetic_eval.jsonl"


def _routine_transcripts() -> list:
    return [
        ("Patient has mild sore throat for 2 days. No fever. Improving with tea and rest.", "routine"),
        ("Caller: My son has a runny nose and mild cough. No fever for 48 hours. He's eating and playing. Just checking if we need to come in.", "routine"),
        ("I've had congestion and a mild headache for five days. No fever, no facial pain. OTC helps a bit.", "routine"),
        ("Mild ear pressure since a flight 4 days ago. No pain, no discharge. Hearing is fine. It's improving.", "routine"),
        ("Hoarse voice for three days after a cold. No trouble swallowing or breathing. Resting voice helps.", "routine"),
        ("Patient reports mild sore throat and swollen glands for three days. No fever. Can swallow fine. A bit better than yesterday.", "routine"),
    ]


def _semi_urgent_transcripts() -> list:
    return [
        ("Sore throat and fever 101 for three days. Getting worse. Can swallow but it hurts. No trouble breathing. Need to be seen soon.", "semi-urgent"),
        ("Bad ear pain and fluid from the ear. Low-grade fever. Two days. Pain is significant. Need an appointment.", "semi-urgent"),
        ("Severe sinus pain and pressure for a week. Getting worse when I bend forward. Congestion and yellowish discharge. No fever. Had something similar before.", "semi-urgent"),
        ("Daughter age 8 has high fever and very sore throat. Drinking but crying when she swallows. No breathing problems. Two days. Need to be seen soon.", "semi-urgent"),
    ]


def _urgent_transcripts() -> list:
    return [
        ("Child has trouble breathing and is making a high-pitched noise when he breathes. Bad sore throat. He's 4. We're on the way to the ER.", "urgent"),
        ("Sudden complete hearing loss in my right ear this morning. No trauma. No sound at all on that side.", "urgent"),
        ("I'm immunocompromised on chemo. Severe sore throat and fever 103. Getting worse. I need to be seen today.", "urgent"),
    ]


def _reference_summary_for(transcript: str, urgency: str) -> str:
    """
    Generate a reference summary that is correct, faithful, and relevant.
    Used as gold standard; model summaries are compared to transcript via metrics.
    """
    t = transcript.lower()
    parts = []
    if "sore throat" in t or "throat" in t:
        parts.append("Patient reports throat symptoms.")
    if "ear" in t:
        parts.append("Ear symptoms noted.")
    if "fever" in t and "no fever" not in t:
        parts.append("Fever present.")
    elif "no fever" in t:
        parts.append("No fever.")
    if "mild" in t:
        parts.append("Mild severity.")
    if "severe" in t:
        parts.append("Severe symptoms.")
    if "improving" in t:
        parts.append("Improving trend.")
    if "worsen" in t or "getting worse" in t:
        parts.append("Worsening trend.")
    if "day" in t or "days" in t or "week" in t:
        parts.append("Duration documented.")
    if "breathing" in t and "no trouble breathing" not in t and "no breathing" not in t:
        parts.append("Breathing difficulty or concern.")
    if "immunocompromised" in t or "chemo" in t:
        parts.append("Immunocompromised patient.")
    if "sudden" in t and "hearing" in t:
        parts.append("Sudden hearing loss.")
    if not parts:
        parts.append("ENT-related symptoms from transcript.")
    parts.append(f"Urgency: {urgency}.")
    return " ".join(parts)


def _unfaithful_summary(transcript: str) -> str:
    """Generate a summary that intentionally adds facts not in transcript (for testing faithfulness)."""
    base = _reference_summary_for(transcript, "routine")
    return base + " Patient has diabetes and high fever."  # Often not in transcript


def _irrelevant_summary() -> str:
    """Summary with off-topic content (for testing relevance)."""
    return "The weather today is fine. Patient likes apples. Scheduling is available next week. Thank you for calling."


def generate_synthetic_eval(
    n_routine: int = 10,
    n_semi: int = 6,
    n_urgent: int = 4,
    include_bad_examples: bool = True,
    out_path: Path | None = None,
) -> str:
    """
    Generate synthetic evaluation JSONL.
    Each line: {"transcript": ..., "reference_summary": ..., "urgency": ..., "expected_urgency": ...}
    If include_bad_examples, adds rows with unfaithful or irrelevant summaries for metric testing.
    """
    out_path = out_path or OUT_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_transcripts = (
        _routine_transcripts()
        + _semi_urgent_transcripts()
        + _urgent_transcripts()
    )
    random.seed(42)
    chosen = []
    for _ in range(n_routine):
        chosen.append(random.choice(_routine_transcripts()))
    for _ in range(n_semi):
        chosen.append(random.choice(_semi_urgent_transcripts()))
    for _ in range(n_urgent):
        chosen.append(random.choice(_urgent_transcripts()))
    random.shuffle(chosen)

    rows = []
    for transcript, urgency in chosen:
        rows.append({
            "transcript": transcript,
            "reference_summary": _reference_summary_for(transcript, urgency),
            "urgency": urgency,
            "expected_urgency": urgency,
        })

    if include_bad_examples:
        t1, u1 = _routine_transcripts()[0]
        rows.append({
            "transcript": t1,
            "reference_summary": _unfaithful_summary(t1),
            "urgency": u1,
            "expected_urgency": u1,
            "note": "unfaithful_reference_for_testing",
        })
        rows.append({
            "transcript": t1,
            "reference_summary": _irrelevant_summary(),
            "urgency": u1,
            "expected_urgency": u1,
            "note": "irrelevant_reference_for_testing",
        })

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    return str(out_path)


if __name__ == "__main__":
    path = generate_synthetic_eval()
    print(f"Wrote synthetic eval data to {path}")
