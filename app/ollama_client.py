import json
import logging
import os
import re

import httpx

from app.config import settings

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
from app.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT_TEMPLATE


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


async def _call_groq(transcript: str, patient_history: dict) -> dict:
    """Call Groq API (OpenAI-compatible) for triage."""
    system_prompt, user_prompt = _build_prompt(transcript, patient_history)
    model = settings.GROQ_MODEL
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
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
                    "temperature": 0.2,
                },
            )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data["choices"][0]["message"]["content"] or ""
        logger.info("[TRIAGE] model=%s | llm_raw_output:\n%s", model, raw_text)
        result = parse_triage_response(raw_text)
        if not result["flags"]:
            result["flags"] = extract_flags_from_transcript(transcript, result["urgency"])
        return result
    except Exception as e:
        print(f"⚠️ Groq API error: {e}")
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
    Call LLM for triage analysis. Uses Groq if LLM_PROVIDER=groq and GROQ_API_KEY is set,
    otherwise uses Ollama.
    
    Returns dict with keys:
    - summary: Clinical summary
    - urgency: Urgency level (routine, semi-urgent, urgent)
    - findings: Key findings/red flags
    - flags: Tagged keywords explaining why AI made this decision
    - reasoning: Explanation for urgency classification
    """
    if patient_history is None:
        patient_history = {}

    # Use Groq if configured
    if getattr(settings, "LLM_PROVIDER", "ollama") == "groq" and getattr(settings, "GROQ_API_KEY", None):
        return await _call_groq(transcript, patient_history)

    # Ollama path
    model = settings.OLLAMA_MODEL_NAME
    logger.info("[TRIAGE] Calling finetuned Ollama model: %s at %s", model, settings.OLLAMA_BASE_URL)
    system_prompt, user_prompt = _build_prompt(transcript, patient_history)
    prompt = system_prompt + "\n\n" + user_prompt
    # #region agent log
    try:
        import json as _json
        import time as _time
        _log_path = "/Users/joshuamatni/Downloads/ai_service/.cursor/debug-891cae.log"
        with open(_log_path, "a") as _f:
            _f.write(_json.dumps({"sessionId":"891cae","hypothesisId":"H1_H2_H4_H5","timestamp":int(_time.time()*1000),"location":"ollama_client.py:call_ollama","message":"Prompt structure before Ollama","data":{"prompt_len":len(prompt),"prompt_first_500":prompt[:500],"prompt_last_600":prompt[-600:],"has_SUMMARY_in_prompt":"SUMMARY:" in prompt,"transcript_in_last_800":transcript[:200] in prompt[-800:] if transcript else False,"transcript_len":len(transcript),"runId":"post-fix"}})+"\n")
    except Exception:
        pass
    # #endregion
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
        _trace("call_ollama", {
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
