import json
import logging
import os
import re

import httpx

from app.config import settings
from app.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT_TEMPLATE,
    REVIEW_SYSTEM_PROMPT,
    REVIEW_USER_PROMPT_TEMPLATE,
    TRIAGE_SYSTEM_PROMPT,
    TRIAGE_USER_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)

DEBUG_URGENCY = os.environ.get("DEBUG_URGENCY", "") == "1"
TRACE_PATH = os.environ.get("URGENCY_TRACE_PATH", "urgency_trace.jsonl")


def _trace(stage: str, data: dict) -> None:
    """Append one JSONL line to trace file when DEBUG_URGENCY=1."""
    if not DEBUG_URGENCY:
        return
    try:
        with open(TRACE_PATH, "a") as f:
            f.write(json.dumps({"stage": stage, **data}) + "\n")
    except Exception:
        pass


# Define keyword dictionaries for automatic flag extraction
KEYWORD_DICTIONARIES = {
    "SYMPTOM": [
        "sore throat", "throat pain", "pharyngeal pain", "pharyngitis",
        "congestion", "nasal congestion", "stuffy nose", "runny nose",
        "cough", "coughing", "dry cough", "productive cough",
        "hoarseness", "voice change", "hoarse voice",
        "post nasal", "post-nasal drip",
        "ear pain", "earache", "otalgia", "otitis",
        "difficulty swallowing", "dysphagia",
        "stridor", "wheezing"
    ],
    "SEVERITY": [
        "mild", "moderate", "severe", "unbearable", "intense",
        "slight", "annoying", "excruciating", "terrible", "worst",
        "sharp", "dull", "constant", "intermittent"
    ],
    "PROGRESSION": [
        "improving", "worsening", "getting worse", "getting better",
        "stable", "unchanged", "progressive", "rapid",
        "yesterday", "better than", "worse than", "trend",
        "deterioration", "improvement"
    ],
    "DURATION": [
        "2 days", "3 days", "1 week", "2 weeks", "days", "week", "weeks",
        "hour", "hours", "this morning", "yesterday", "since",
        "about", "approximately", "for"
    ],
    "RED_FLAG": [
        "breathing difficulty", "difficulty breathing", "shortness of breath",
        "stridor", "wheezing", "respiratory", "airway",
        "severe pain", "unbearable pain",
        "sudden hearing loss", "hearing change",
        "severe dizziness", "vertigo", "fainting",
        "fever", "high temperature",
        "immunocompromised", "immunosuppressed", "AIDS", "HIV",
        "spreading infection", "infected", "infection"
    ],
    "RELIEVING_FACTORS": [
        "warm tea", "tea", "honey", "lozenges", "rest", "sleep",
        "pain relief", "ibuprofen", "acetaminophen", "tylenol",
        "helps", "better with", "improves with"
    ],
    "AGGRAVATING_FACTORS": [
        "cold water", "swallowing", "talking", "shouting",
        "worse with", "aggravated by", "makes worse",
        "exacerbated"
    ],
    "ASSOCIATED_SYMPTOMS": [
        "nasal congestion", "runny nose", "sinus", "sinusitis",
        "headache", "body aches", "muscle aches",
        "fatigue", "tired", "weakness",
        "nausea", "vomiting"
    ],
    "MEDICAL_HISTORY": [
        "diabetes", "diabetic", "hypertension", "high blood pressure",
        "asthma", "immunocompromised", "cancer", "chronic disease",
        "heart disease", "previous", "history of"
    ]
}


def _build_prompt(transcript: str, patient_history: dict) -> tuple[str, str]:
    """Build system and user prompts for triage. Returns (system_prompt, user_prompt)."""
    medical_history = ", ".join(patient_history.get("medicalHistory", [])) or "None documented"
    previous_visits = ", ".join(patient_history.get("previousVisits", [])) or "None documented"
    allergies = ", ".join(patient_history.get("allergies", [])) or "None documented"
    user_prompt = (
        TRIAGE_USER_PROMPT_TEMPLATE
        .replace("<<TRANSCRIPT>>", transcript)
        .replace("<<PATIENT_HISTORY>>", medical_history)
        .replace("<<PREVIOUS_VISITS>>", previous_visits)
        .replace("<<ALLERGIES>>", allergies)
    )
    return TRIAGE_SYSTEM_PROMPT.strip(), user_prompt


def _build_judge_prompt(
    transcript: str,
    patient_history: dict,
    ollama_result: dict,
    rf_result: dict,
) -> tuple[str, str]:
    """Build Groq judge prompts from transcript + model outputs."""
    history = ", ".join(patient_history.get("medicalHistory", [])) or "None documented"
    findings = ollama_result.get("findings", [])
    findings_text = "; ".join(findings) if findings else "None"
    user_prompt = (
        JUDGE_USER_PROMPT_TEMPLATE
        .replace("<<TRANSCRIPT>>", transcript)
        .replace("<<PATIENT_HISTORY>>", history)
        .replace("<<OLLAMA_SUMMARY>>", ollama_result.get("summary", ""))
        .replace("<<OLLAMA_FINDINGS>>", findings_text)
        .replace("<<OLLAMA_URGENCY>>", ollama_result.get("urgency", "routine"))
        .replace("<<OLLAMA_REASONING>>", ollama_result.get("reasoning", ""))
        .replace("<<RF_URGENCY>>", rf_result.get("urgency", "routine"))
        .replace("<<RF_CONFIDENCE>>", str(rf_result.get("confidence", 0.0)))
        .replace("<<RF_SOURCE>>", rf_result.get("source", "random_forest"))
    )
    return JUDGE_SYSTEM_PROMPT.strip(), user_prompt


def _build_review_prompt(transcript: str, patient_history: dict, ollama_result: dict) -> tuple[str, str]:
    """Build Groq output-review prompts (completeness vs transcript)."""
    medical_history = ", ".join(patient_history.get("medicalHistory", [])) or "None documented"
    previous_visits = ", ".join(patient_history.get("previousVisits", [])) or "None documented"
    allergies = ", ".join(patient_history.get("allergies", [])) or "None documented"
    history_block = (
        f"Medical history: {medical_history}\n"
        f"Previous ENT visits: {previous_visits}\n"
        f"Allergies: {allergies}"
    )
    findings = ollama_result.get("findings", [])
    findings_text = "\n".join(f"- {f}" for f in findings) if findings else "(none listed)"
    user_prompt = (
        REVIEW_USER_PROMPT_TEMPLATE.replace("<<TRANSCRIPT>>", transcript)
        .replace("<<PATIENT_HISTORY>>", history_block)
        .replace("<<OLLAMA_SUMMARY>>", ollama_result.get("summary", ""))
        .replace("<<OLLAMA_FINDINGS>>", findings_text)
        .replace("<<OLLAMA_URGENCY>>", ollama_result.get("urgency", "routine"))
        .replace("<<OLLAMA_REASONING>>", ollama_result.get("reasoning", ""))
    )
    return REVIEW_SYSTEM_PROMPT.strip(), user_prompt


async def _post_groq_chat(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    timeout: float = 60.0,
    temperature: float = 0.2,
) -> str:
    """POST to Groq OpenAI-compatible chat; returns assistant message text or empty string if misconfigured."""
    if not getattr(settings, "GROQ_API_KEY", None):
        return ""
    model = settings.GROQ_MODEL
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        logger.warning("[GROQ] chat request failed: %s", e)
        return ""


def _llm_config_error(message: str) -> dict:
    """Structured triage dict when the LLM cannot run (misconfiguration or disabled path)."""
    logger.error("[TRIAGE] %s", message)
    return {
        "summary": f"Error: {message}",
        "urgency": "routine",
        "findings": [],
        "flags": [],
        "reasoning": "Error during processing",
    }


async def call_groq_judge(
    transcript: str,
    patient_history: dict,
    ollama_result: dict,
    rf_result: dict,
) -> dict:
    """Call Groq LLM-as-a-judge to resolve urgency disagreement."""
    if not getattr(settings, "GROQ_API_KEY", None):
        return {
            "urgency": "urgent",
            "reasoning": "Judge unavailable: GROQ_API_KEY is not set. Escalating safely.",
            "decision_factors": "judge_unavailable,safety_first",
            "raw_output": "",
        }
    system_prompt, user_prompt = _build_judge_prompt(
        transcript=transcript,
        patient_history=patient_history,
        ollama_result=ollama_result,
        rf_result=rf_result,
    )
    raw_text = await _post_groq_chat(
        system_prompt, user_prompt, max_tokens=300, timeout=60.0, temperature=0.0
    )
    if not raw_text:
        return {
            "urgency": "urgent",
            "reasoning": "Judge unavailable: empty Groq response. Escalating safely.",
            "decision_factors": "judge_unavailable,safety_first",
            "raw_output": "",
        }
    logger.info("[JUDGE] model=%s | raw_output:\n%s", settings.GROQ_MODEL, raw_text)
    parsed = parse_judge_response(raw_text)
    parsed["raw_output"] = raw_text
    return parsed


def parse_review_response(response_text: str, original_summary: str) -> dict:
    """Parse Groq output-review schema; returns applied, revised_summary, coverage_ok, missing_or_omitted."""
    text = (response_text or "").strip()
    out: dict = {
        "applied": False,
        "revised_summary": "",
        "coverage_ok": True,
        "missing_or_omitted": "",
        "raw_output": text,
    }
    cov_m = re.search(r"COVERAGE_OK:\s*(yes|no)\b", text, re.IGNORECASE)
    missing_m = re.search(r"MISSING_OR_OMITTED:\s*(.+?)(?=REVISED_SUMMARY:|$)", text, re.DOTALL | re.IGNORECASE)
    rev_m = re.search(r"REVISED_SUMMARY:\s*(.+)$", text, re.DOTALL | re.IGNORECASE)

    coverage_ok = True
    if cov_m:
        coverage_ok = cov_m.group(1).lower().strip() == "yes"
    out["coverage_ok"] = coverage_ok
    if missing_m:
        out["missing_or_omitted"] = missing_m.group(1).strip()

    revised = rev_m.group(1).strip() if rev_m else ""
    out["revised_summary"] = revised
    if not revised:
        return out
    rev_norm = revised.upper().replace(" ", "_")
    if "USE_ORIGINAL" in rev_norm:
        return out
    if coverage_ok:
        return out
    # Gaps reported: apply revised summary when it looks substantive
    if len(revised) < 20:
        return out
    out["applied"] = True
    return out


async def call_groq_output_review(
    transcript: str,
    patient_history: dict,
    ollama_result: dict,
) -> dict:
    """Groq pass: check triage summary/findings vs transcript; optionally revise summary."""
    if not getattr(settings, "ENABLE_LLM_OUTPUT_REVIEW", True):
        return {
            "applied": False,
            "revised_summary": "",
            "coverage_ok": True,
            "missing_or_omitted": "",
            "raw_output": "",
        }
    if not getattr(settings, "GROQ_API_KEY", None):
        logger.info("[REVIEW] skipped: GROQ_API_KEY not set")
        return {
            "applied": False,
            "revised_summary": "",
            "coverage_ok": True,
            "missing_or_omitted": "",
            "raw_output": "",
        }
    system_prompt, user_prompt = _build_review_prompt(transcript, patient_history, ollama_result)
    raw_text = await _post_groq_chat(system_prompt, user_prompt, max_tokens=700, timeout=45.0)
    if not raw_text:
        return {
            "applied": False,
            "revised_summary": "",
            "coverage_ok": True,
            "missing_or_omitted": "",
            "raw_output": "",
        }
    logger.info("[REVIEW] model=%s | raw_output:\n%s", settings.GROQ_MODEL, raw_text)
    original = ollama_result.get("summary", "") or ""
    parsed = parse_review_response(raw_text, original)
    if parsed["applied"] and parsed.get("revised_summary"):
        logger.info(
            "[REVIEW] summary revised | missing_or_omitted=%r",
            (parsed.get("missing_or_omitted") or "")[:300],
        )
    return parsed


async def apply_groq_output_review_if_enabled(
    transcript: str,
    patient_history: dict,
    llm_result: dict,
) -> None:
    """Mutate llm_result['summary'] when Groq review finds gaps (no-op if disabled or unchanged)."""
    review = await call_groq_output_review(transcript, patient_history, llm_result)
    if review.get("applied") and review.get("revised_summary"):
        llm_result["summary"] = review["revised_summary"]


async def _call_ollama_local(transcript: str, patient_history: dict) -> dict:
    """Call local Ollama HTTP /api/generate (same prompt shape as before Hub inference)."""
    model = settings.OLLAMA_MODEL_NAME
    logger.info("[TRIAGE] Calling Ollama model: %s at %s", model, settings.OLLAMA_BASE_URL)
    system_prompt, user_prompt = _build_prompt(transcript, patient_history)
    prompt = system_prompt + "\n\n" + user_prompt
    payload = {
        "model": settings.OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("response", "")
        logger.info("[TRIAGE] model=%s | llm_raw_output:\n%s", model, raw_text)
        result = parse_triage_response(raw_text)
        _trace("call_ollama_local", {
            "llm_raw_preview": raw_text[:500] if raw_text else "",
            "parsed_urgency": result.get("urgency"),
        })
        if not result["flags"]:
            result["flags"] = extract_flags_from_transcript(transcript, result["urgency"])
        return result
    except httpx.TimeoutException:
        print("⚠️ Ollama timeout - returning default response")
        return {
            "summary": "Patient presents with ENT-related symptoms. Further assessment needed.",
            "urgency": "routine",
            "findings": [],
            "flags": [],
            "reasoning": "Default due to timeout",
        }
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return {
            "summary": f"Error: {str(e)}",
            "urgency": "routine",
            "findings": [],
            "flags": [],
            "reasoning": "Error during processing",
        }


async def call_ollama(
    transcript: str,
    patient_history: dict = None
) -> dict:
    """
    Call LLM for triage analysis.

    Triage generation always uses Ollama in the consensus pipeline.

    Returns dict with keys:
    - summary: Clinical summary
    - urgency: Urgency level (routine, semi-urgent, urgent)
    - findings: Key findings/red flags
    - flags: Tagged keywords explaining why AI made this decision
    - reasoning: Explanation for urgency classification
    """
    if patient_history is None:
        patient_history = {}

    provider = (getattr(settings, "LLM_PROVIDER", None) or "ollama").strip().lower()
    if provider != "ollama":
        logger.warning(
            "[TRIAGE] LLM_PROVIDER=%r requested, but triage generation is pinned to Ollama in this pipeline.",
            settings.LLM_PROVIDER,
        )
    return await _call_ollama_local(transcript, patient_history)


def parse_judge_response(response_text: str) -> dict:
    """Parse LLM-as-a-judge output schema."""
    result = {
        "urgency": "urgent",
        "reasoning": "Judge output malformed. Escalating safely.",
        "decision_factors": "malformed_output,safety_first",
    }
    text = response_text or ""
    urgency_match = re.search(r"FINAL_URGENCY:\s*(routine|semi-urgent|urgent)", text, re.IGNORECASE)
    reasoning_match = re.search(r"JUDGE_REASONING:\s*(.+?)(?=DECISION_FACTORS:|$)", text, re.DOTALL | re.IGNORECASE)
    factors_match = re.search(r"DECISION_FACTORS:\s*(.+)$", text, re.DOTALL | re.IGNORECASE)
    if urgency_match:
        result["urgency"] = urgency_match.group(1).lower().strip()
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()
    if factors_match:
        result["decision_factors"] = factors_match.group(1).strip()
    return result


def parse_triage_response(response_text: str) -> dict:
    """
    Parse the LLM response into structured fields.
    
    Expected format:
    SUMMARY: [text]
    FINDINGS: [text]
    FLAGS: [TAG] keyword, [TAG] keyword, etc.
    URGENCY: [routine/semi-urgent/urgent]
    REASONING: [text]
    """
    
    result = {
        "summary": "",
        "findings": [],
        "flags": [],
        "urgency": "routine",
        "reasoning": ""
    }
    
    try:
        # Extract SUMMARY
        summary_match = re.search(r'SUMMARY:\s*(.+?)(?=FINDINGS:|FLAGS:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if summary_match:
            result["summary"] = summary_match.group(1).strip()
        
        # Extract FINDINGS
        findings_match = re.search(r'FINDINGS:\s*(.+?)(?=FLAGS:|URGENCY:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if findings_match:
            findings_text = findings_match.group(1).strip()
            # Split by newlines and clean up
            result["findings"] = [f.strip() for f in findings_text.split('\n') if f.strip() and f.strip() != "**"]
        
        # Extract FLAGS with tags
        flags_match = re.search(r'FLAGS:\s*(.+?)(?=URGENCY:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if flags_match:
            flags_text = flags_match.group(1).strip()
            result["flags"] = parse_flags(flags_text)
        
        # Extract URGENCY
        urgency_match = re.search(r'URGENCY:\s*(\w+(?:-\w+)?)', response_text, re.IGNORECASE)
        if urgency_match:
            urgency = urgency_match.group(1).lower()
            if urgency in ["routine", "semi-urgent", "urgent"]:
                result["urgency"] = urgency
        _trace("parse_triage_response", {
            "urgency_matched": bool(urgency_match),
            "extracted_urgency": result["urgency"],
        })

        # Extract REASONING
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n\n|$)', response_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()

        # Minimal fallback when model skipped SUMMARY: label
        text = (response_text or "").strip()
        if not result["summary"]:
            # Fallback 1: model output "Patient presents with..." as first line (no SUMMARY: label)
            patient_line = re.search(
                r'(Patient presents with [^.]+(?:\. [^.]+)*\.)',
                text,
                re.IGNORECASE | re.DOTALL
            )
            if patient_line and len(patient_line.group(1)) > 30:
                result["summary"] = patient_line.group(1).strip()[:800]
            # Fallback 2: build from FINDINGS only (safe, no REASONING - can contradict transcript)
            elif result["findings"]:
                result["summary"] = "Patient presents with " + ". ".join(result["findings"][:5]) + "."

        if not result["summary"] or not summary_match:
            logger.warning(
                "[PARSE] SUMMARY: missing in model output | first_150_chars=%r",
                text[:150],
            )

        return result

    except Exception as e:
        print(f"⚠️ Failed to parse LLM response: {e}")
        # Return best-effort parsing
        return result


def parse_flags(flags_text: str) -> list:
    """
    Parse flags from format: [TAG] keyword, [TAG] keyword, etc.
    
    Returns list of dicts with structure:
    [
        {"tag": "SYMPTOM", "keyword": "sore throat"},
        {"tag": "SEVERITY", "keyword": "mild"},
        ...
    ]
    """
    flags = []
    
    # Match pattern: [TAG] keyword (with comma separation)
    # Regex: \[([A-Z_]+)\]\s+([^,\[\]]+)
    pattern = r'\[([A-Z_]+)\]\s+([^,\[\]]+)'
    matches = re.findall(pattern, flags_text)
    
    for tag, keyword in matches:
        flags.append({
            "tag": tag.strip(),
            "keyword": keyword.strip()
        })
    
    return flags


def extract_flags_from_transcript(transcript: str, urgency: str) -> list:
    """
    Automatically extract flags from transcript by matching keywords.
    This is a fallback when the LLM doesn't provide FLAGS.
    """
    flags = []
    transcript_lower = transcript.lower()
    
    # Track which keywords we've already added to avoid duplicates
    added_keywords = set()
    
    # Search through each category
    for tag, keywords in KEYWORD_DICTIONARIES.items():
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Check if keyword appears in transcript
            if keyword_lower in transcript_lower:
                # Avoid adding duplicate keywords
                if keyword_lower not in added_keywords:
                    flags.append({
                        "tag": tag,
                        "keyword": keyword
                    })
                    added_keywords.add(keyword_lower)
    
    # If no flags found, add some basic ones based on urgency
    if not flags:
        if urgency == "urgent":
            flags.append({"tag": "RED_FLAG", "keyword": "critical symptoms detected"})
        elif urgency == "semi-urgent":
            flags.append({"tag": "SEVERITY", "keyword": "moderate severity"})
        else:
            flags.append({"tag": "SEVERITY", "keyword": "mild symptoms"})
    
    return flags


# RED FLAG KEYWORDS - Auto-escalate to URGENT
CRITICAL_RED_FLAGS = [
    "breathing difficulty", "difficulty breathing", "shortness of breath", "breath shortness",
    "stridor", "wheezing", "wheeze", "respiratory distress",
    "airway", "choking", "asphyxia",
    "severe throat pain", "severe pain", "unbearable pain", "excruciating",
    "dysphagia", "difficulty swallowing", "cannot swallow",
    "severe dizziness", "vertigo", "fainting", "syncope",
    "sudden hearing loss", "hearing loss", "deafness",
    "fever", "high temperature", "chills",
    "immunocompromised", "immunosuppressed", "AIDS", "HIV",
    "spreading infection", "infected", "sepsis",
    "facial swelling", "throat swelling", "edema",
    "severe bleeding", "hemorrhage", "blood",
]

# MEDICAL HISTORY RISK FLAGS - Escalate one level
MEDICAL_RISK_FACTORS = {
    "immunocompromised": ["HIV", "AIDS", "cancer", "immunosuppressant", "leukemia", "lymphoma"],
    "diabetes": ["diabetes", "diabetic"],
    "lung_disease": ["asthma", "COPD", "emphysema", "chronic bronchitis"],
    "heart_disease": ["heart", "cardiac", "arrhythmia", "hypertension"],
    "previous_complications": ["previous ENT surgery", "recurrent", "chronic infection"],
}


def validate_urgency_classification(
    transcript: str,
    llm_urgency: str,
    flags: list,
    patient_history: dict = None,
    ml_confidence: float = 0.0,
    summary: str = "",
) -> tuple:
    """
    Secondary validation layer to ensure urgency classification is correct.
    Can escalate (e.g. red flags) or downgrade (e.g. LLM says urgent but evidence is routine).

    Returns:
        tuple: (adjusted_urgency, confidence_level)
    """
    if getattr(settings, "TRUST_LLM_URGENCY", False):
        return (llm_urgency, "high")
    transcript_lower = transcript.lower()
    summary_lower = (summary or "").lower()
    
    # Check for critical red flags - auto-escalate to URGENT
    for red_flag in CRITICAL_RED_FLAGS:
        if red_flag.lower() in transcript_lower:
            # print(f"⚠️ CRITICAL RED FLAG DETECTED: {red_flag} → Escalating to URGENT")
            return ("urgent", "high")
    
    # Check for red flag tags in the flags list
    has_red_flag = any(f.get("tag") == "RED_FLAG" for f in flags)
    
    # Check patient medical history for risk factors
    patient_history = patient_history or {}
    medical_history = (patient_history.get("medicalHistory") or [])
    medical_history_str = " ".join(medical_history).lower()
    
    is_immunocompromised = any(
        risk in medical_history_str 
        for risk in MEDICAL_RISK_FACTORS["immunocompromised"]
    )
    is_high_risk = any(
        risk in medical_history_str 
        for risks in MEDICAL_RISK_FACTORS.values() 
        for risk in risks
    )
    
    # Decision logic for urgency adjustment
    adjusted_urgency = llm_urgency
    confidence = "high"
    
    # Rule 1: Red flags present = URGENT
    # Use "red flag" or "red-flag" only; do NOT use "red_flag" (matches slot "red_flags")
    red_flag_in_transcript = "red flag" in transcript_lower or "red-flag" in transcript_lower
    rule_fired = "none"
    # Trust LLM when it says routine and evidence clearly supports mild+improving (e.g. ear infection with discharge, improving)
    routine_indicators = (
        ("mild" in transcript_lower or "mild" in summary_lower)
        and ("improving" in transcript_lower or "improving" in summary_lower or "routine" in summary_lower)
    )
    if (has_red_flag or red_flag_in_transcript) and not (llm_urgency == "routine" and routine_indicators):
        adjusted_urgency = "urgent"
        confidence = "high"
        rule_fired = "1"

    # Rule 2: Immunocompromised + any symptoms = at least SEMI-URGENT
    elif is_immunocompromised:
        rule_fired = "2"
        if adjusted_urgency == "routine":
            adjusted_urgency = "semi-urgent"
            confidence = "medium"
            # print(f"⚠️ Immunocompromised patient with symptoms → Escalating to SEMI-URGENT")
        elif adjusted_urgency == "semi-urgent":
            confidence = "high"

    # Rule 3: High-risk patient + worsening symptoms = at least SEMI-URGENT
    elif is_high_risk:
        rule_fired = "3"
        worsening = any(
            word in transcript_lower 
            for word in ["worsening", "worse", "getting worse", "deteriorating", "rapid"]
        )
        if worsening and adjusted_urgency == "routine":
            adjusted_urgency = "semi-urgent"
            confidence = "medium"
            # print(f"⚠️ High-risk patient with worsening symptoms → Escalating to SEMI-URGENT")

    # Rule 3b: Downgrade when LLM says urgent/semi-urgent but evidence clearly indicates routine
    elif adjusted_urgency in ("urgent", "semi-urgent") and not has_red_flag and not is_immunocompromised:
        routine_indicators = (
            "mild" in transcript_lower or "mild" in summary_lower
        ) and (
            "improving" in transcript_lower or "improving" in summary_lower or
            "getting better" in transcript_lower or "better" in summary_lower or
            "stable" in transcript_lower or "stable" in summary_lower
        ) and (
            "no red flags" in transcript_lower or "no red flags" in summary_lower or
            "red_flags:no" in transcript_lower or "red_flags:none" in transcript_lower
        )
        if routine_indicators:
            rule_fired = "3b"
            adjusted_urgency = "routine"
            confidence = "high"
            # print(f"✓ Downgrading {llm_urgency} → routine: transcript/summary indicate mild, improving, no red flags")

    # Rule 3c: semi-urgent without transcript support (avoid using semi as a default tier)
    if (
        adjusted_urgency == "semi-urgent"
        and not has_red_flag
        and not red_flag_in_transcript
        and not is_immunocompromised
        and not is_high_risk
    ):
        worsen_phrases = (
            "worsening",
            "getting worse",
            "deteriorating",
            "progressively",
            "much worse",
            "spreading",
        )
        has_worsening = any(p in transcript_lower for p in worsen_phrases)
        # Standalone "worse" but not "not worse" / "no worse"
        has_worsening = has_worsening or bool(
            re.search(r"(?<!not )(?<!no )\bworse\b", transcript_lower)
        )
        has_worsening = has_worsening or bool(
            re.search(r"\brapid(?:ly)?\b", transcript_lower)
        )
        severe_words = (
            "severe",
            "unbearable",
            "worst",
            "excruciating",
            "high fever",
            "103°",
            "104°",
            "105°",
        )
        has_severe = any(w in transcript_lower for w in severe_words)
        mildish = "mild" in transcript_lower or "mild" in summary_lower
        same_as_before = bool(re.search(r"\bsame\b", transcript_lower))
        moderate_stable = (
            ("moderate" in transcript_lower or "moderate" in summary_lower)
            and (
                "stable" in transcript_lower
                or "stable" in summary_lower
                or "unchanged" in transcript_lower
                or same_as_before
            )
        )
        if not has_worsening and not has_severe and (mildish or moderate_stable):
            adjusted_urgency = "routine"
            confidence = "medium"
            rule_fired = "3c"

    # Rule 4: Low ML confidence on serious classification = reduce confidence
    if ml_confidence < 0.6 and adjusted_urgency in ["semi-urgent", "urgent"]:
        confidence = "medium"
    
    # Rule 5: Multiple severe indicators = high confidence
    severe_indicators = sum([
        has_red_flag,
        is_immunocompromised,
        is_high_risk,
        "severe" in transcript_lower,
        "worsening" in transcript_lower,
    ])
    if severe_indicators >= 2:
        confidence = "high"
    
    _trace("validate_urgency_classification", {
        "rule_fired": rule_fired,
        "llm_urgency": llm_urgency,
        "adjusted_urgency": adjusted_urgency,
        "has_red_flag": has_red_flag,
        "red_flag_in_transcript": red_flag_in_transcript,
    })
    # print(f"✓ Urgency validation: {llm_urgency} → {adjusted_urgency} (confidence: {confidence})")
    return (adjusted_urgency, confidence)
