import httpx
import re
from app.config import settings
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


async def call_ollama(
    transcript: str,
    patient_history: dict = None
) -> dict:
    """
    Call Ollama LLM for triage analysis.
    
    Returns dict with keys:
    - summary: Clinical summary
    - urgency: Urgency level (routine, semi-urgent, urgent)
    - findings: Key findings/red flags
    - flags: Tagged keywords explaining why AI made this decision
    - reasoning: Explanation for urgency classification
    """
    
    if patient_history is None:
        patient_history = {}
    
    # Format patient history for the prompt
    medical_history = ", ".join(patient_history.get("medicalHistory", [])) or "None documented"
    previous_visits = ", ".join(patient_history.get("previousVisits", [])) or "None documented"
    allergies = ", ".join(patient_history.get("allergies", [])) or "None documented"
    
    # Build the prompt with patient history
    user_prompt = (
        TRIAGE_USER_PROMPT_TEMPLATE
        .replace("<<TRANSCRIPT>>", transcript)
        .replace("<<PATIENT_HISTORY>>", medical_history)
        .replace("<<PREVIOUS_VISITS>>", previous_visits)
        .replace("<<ALLERGIES>>", allergies)
    )
    
    prompt = (
        TRIAGE_SYSTEM_PROMPT.strip() +
        "\n\n" +
        user_prompt
    )

    payload = {
        "model": settings.OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            headers = {"Content-Type": "application/json"}

            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json=payload,
                headers=headers
            )

        resp.raise_for_status()

        data = resp.json()
        raw_text = data.get("response", "")

        # Parse the structured response
        result = parse_triage_response(raw_text)
        
        # If flags are empty, extract them automatically from transcript
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
            "reasoning": "Default due to timeout"
        }
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return {
            "summary": f"Error: {str(e)}",
            "urgency": "routine",
            "findings": [],
            "flags": [],
            "reasoning": "Error during processing"
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
