"""
Rule-based metrics for validating LLM triage summaries.
Scores: correctness, faithfulness, relevance (no LLM-as-judge).
"""

import re
from dataclasses import dataclass
from typing import List, Set, Tuple


@dataclass
class ValidationScores:
    correctness: float  # 0-1
    faithfulness: float  # 0-1
    relevance: float  # 0-1
    details: dict  # optional per-metric details


# ENT-relevant terms for relevance scoring
ENT_SYMPTOM_TERMS = {
    "sore throat", "throat", "pharyngitis", "congestion", "nasal", "sinus",
    "ear", "earache", "otalgia", "hearing", "hoarse", "voice", "cough",
    "swallow", "dysphagia", "stridor", "wheezing", "breathing", "pain",
    "fever", "discharge", "pressure", "dizziness", "vertigo", "symptom",
    "duration", "days", "weeks", "improving", "worsening", "severe", "mild",
    "moderate", "red flag", "urgent", "routine", "findings", "patient",
}
# Negation + key fact patterns for correctness
NEGATION_PATTERNS = re.compile(
    r"\b(no|not|without|denies|none)\s+(fever|pain|discharge|breathing difficulty|dysphagia)\b",
    re.I,
)
DURATION_PATTERN = re.compile(r"\b(\d+)\s*(day|days|week|weeks|hour|hours)\b", re.I)
SEVERITY_PATTERN = re.compile(r"\b(mild|moderate|severe|improving|worsening|stable)\b", re.I)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _tokenize(text: str) -> Set[str]:
    normalized = _normalize(text)
    # Keep alphanumeric and apostrophe (e.g. "don't")
    tokens = set(re.findall(r"[a-z0-9']+", normalized))
    return tokens


def _extract_key_facts(transcript: str) -> dict:
    """Extract duration, severity, and negation facts from transcript."""
    t = _normalize(transcript)
    facts = {
        "durations": set(DURATION_PATTERN.findall(t)),
        "severity_terms": set(SEVERITY_PATTERN.findall(t)),
        "negations": set(),  # (negation, term) e.g. ("no", "fever")
    }
    for m in NEGATION_PATTERNS.finditer(t):
        facts["negations"].add((m.group(1).lower(), m.group(2).lower()))
    return facts


def score_correctness(transcript: str, summary: str) -> Tuple[float, dict]:
    """
    Score whether the summary is factually consistent with the transcript.
    Checks: no contradiction of negations (e.g. transcript says 'no fever', summary must not say 'fever');
    key facts (duration, severity) appear or are not contradicted.
    Returns (0-1 score, details dict).
    """
    details = {"contradictions": 0, "fact_checks": 0, "passed": 0}
    t = _normalize(transcript)
    s = _normalize(summary)
    facts = _extract_key_facts(transcript)

    # Contradiction: transcript says "no X", summary says "X"
    for neg, term in facts["negations"]:
        details["fact_checks"] += 1
        # Summary should not assert the negated term as present
        if re.search(r"\b" + term + r"\b", s) and "no " + term not in s and "without " + term not in s:
            details["contradictions"] += 1

    # If transcript has duration/severity, summary ideally reflects it (soft check)
    if facts["durations"]:
        details["fact_checks"] += 1
        if any(d[0] in s and d[1] in s for d in facts["durations"]):
            details["passed"] += 1
    if facts["severity_terms"]:
        details["fact_checks"] += 1
        if any(sev in s for sev in facts["severity_terms"]):
            details["passed"] += 1

    total_checks = details["fact_checks"] + len(facts["negations"])
    if total_checks == 0:
        score = 1.0  # Nothing to check
    else:
        penalties = details["contradictions"] * 0.5  # Heavy penalty for contradiction
        score = max(0.0, 1.0 - penalties - (details["fact_checks"] - details["passed"]) * 0.1)
    details["score_raw"] = score
    return (min(1.0, max(0.0, score)), details)


def score_faithfulness(transcript: str, summary: str) -> Tuple[float, dict]:
    """
    Score whether the summary is grounded in the transcript (no hallucination).
    Uses token overlap: proportion of summary content that appears in transcript.
    Returns (0-1 score, details dict).
    """
    t_tok = _tokenize(transcript)
    s_tok = _tokenize(summary)
    if not s_tok:
        return (1.0, {"overlap_ratio": 1.0, "summary_tokens": 0})
    overlap = len(s_tok & t_tok) / len(s_tok)
    # Slight boost if transcript is long (more chance of overlap)
    score = min(1.0, overlap * 1.1) if overlap > 0.3 else overlap
    return (score, {"overlap_ratio": round(overlap, 3), "summary_tokens": len(s_tok)})


def score_relevance(summary: str) -> Tuple[float, dict]:
    """
    Score whether the summary is relevant to ENT triage (symptoms, severity, urgency).
    Based on presence of ENT-relevant terms and absence of long off-topic content.
    Returns (0-1 score, details dict).
    """
    s = _normalize(summary)
    s_tok = _tokenize(summary)
    if not s_tok:
        return (0.0, {"ent_terms": 0, "total_tokens": 0})
    found = sum(1 for term in ENT_SYMPTOM_TERMS if term in s)
    # Score: proportion of distinct ENT terms present, capped by coverage
    term_ratio = min(1.0, found / 8)  # 8+ distinct ENT terms -> 1.0
    length_ok = 10 <= len(s_tok) <= 150  # Reasonable summary length
    score = term_ratio * 0.85 + (0.15 if length_ok else 0)
    return (min(1.0, score), {"ent_terms_found": found, "total_tokens": len(s_tok)})


def validate_summary(transcript: str, summary: str) -> ValidationScores:
    """
    Run all three metrics and return a ValidationScores object.
    """
    cor, cor_d = score_correctness(transcript, summary)
    faith, faith_d = score_faithfulness(transcript, summary)
    rel, rel_d = score_relevance(summary)
    return ValidationScores(
        correctness=round(cor, 3),
        faithfulness=round(faith, 3),
        relevance=round(rel, 3),
        details={"correctness": cor_d, "faithfulness": faith_d, "relevance": rel_d},
    )
