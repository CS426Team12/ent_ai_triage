"""
Unit tests for app.ollama_client: parse_triage_response, parse_flags,
extract_flags_from_transcript, validate_urgency_classification.
"""
import pytest
from app.ollama_client import (
    parse_triage_response,
    parse_flags,
    extract_flags_from_transcript,
    validate_urgency_classification,
)


class TestParseTriageResponse:
    """Tests for parse_triage_response."""

    def test_parses_full_response(self):
        text = """
SUMMARY: Patient has mild sore throat for 3 days.
FINDINGS: Mild sore throat. No fever.
FLAGS: [SYMPTOM] sore throat, [SEVERITY] mild
URGENCY: routine
REASONING: No red flags, can wait.
"""
        result = parse_triage_response(text)
        assert "sore throat" in result["summary"].lower() or "mild" in result["summary"].lower()
        assert len(result["findings"]) >= 1
        assert len(result["flags"]) >= 1
        assert result["urgency"] == "routine"
        assert "red" in result["reasoning"].lower() or "wait" in result["reasoning"].lower()

    def test_returns_defaults_on_empty(self):
        result = parse_triage_response("")
        assert result["summary"] == ""
        assert result["findings"] == []
        assert result["flags"] == []
        assert result["urgency"] == "routine"
        assert result["reasoning"] == ""

    def test_parses_urgency_semi_urgent(self):
        text = "SUMMARY: x\nFINDINGS: x\nFLAGS: x\nURGENCY: semi-urgent\nREASONING: x"
        result = parse_triage_response(text)
        assert result["urgency"] == "semi-urgent"

    def test_parses_urgency_urgent(self):
        text = "SUMMARY: x\nFINDINGS: x\nFLAGS: x\nURGENCY: urgent\nREASONING: x"
        result = parse_triage_response(text)
        assert result["urgency"] == "urgent"

    def test_missing_urgency_returns_default_routine(self):
        text = "SUMMARY: Some text.\nFINDINGS: x\nFLAGS: x\nREASONING: x"
        result = parse_triage_response(text)
        assert result["urgency"] == "routine"

    def test_multiline_urgency(self):
        text = """
SUMMARY: Patient has mild symptoms.
FINDINGS:
- Finding one
- Finding two
FLAGS: [SYMPTOM] cough
URGENCY: semi-urgent
REASONING: Worsening trend.
"""
        result = parse_triage_response(text)
        assert result["urgency"] == "semi-urgent"
        assert len(result["findings"]) >= 1


class TestParseFlags:
    """Tests for parse_flags."""

    def test_parses_tag_keyword_pairs(self):
        text = "[SYMPTOM] sore throat, [SEVERITY] mild"
        result = parse_flags(text)
        assert len(result) >= 1
        for f in result:
            assert "tag" in f
            assert "keyword" in f

    def test_parses_multiple(self):
        text = "[SYMPTOM] cough, [DURATION] 3 days"
        result = parse_flags(text)
        assert len(result) == 2
        tags = {f["tag"] for f in result}
        assert "SYMPTOM" in tags
        assert "DURATION" in tags

    def test_empty_string_returns_empty_list(self):
        assert parse_flags("") == []


class TestExtractFlagsFromTranscript:
    """Tests for extract_flags_from_transcript."""

    def test_extracts_known_keywords(self):
        transcript = "Patient has sore throat and mild pain for 3 days."
        result = extract_flags_from_transcript(transcript, "routine")
        assert len(result) >= 1
        keywords = [f["keyword"] for f in result]
        assert any("sore throat" in k or "mild" in k or "3 days" in k for k in keywords)

    def test_fallback_for_urgent_when_no_keywords(self):
        result = extract_flags_from_transcript("nothing special here", "urgent")
        assert len(result) >= 1
        assert any(f["tag"] == "RED_FLAG" for f in result)

    def test_fallback_for_routine_when_no_keywords(self):
        result = extract_flags_from_transcript("nothing special", "routine")
        assert len(result) >= 1


class TestValidateUrgencyClassification:
    """Tests for validate_urgency_classification."""

    def test_critical_red_flag_escalates_to_urgent(self):
        urgency, conf = validate_urgency_classification(
            transcript="Patient has difficulty breathing.",
            llm_urgency="routine",
            flags=[],
        )
        assert urgency == "urgent"
        assert conf == "high"

    def test_routine_stays_routine_with_no_flags(self):
        urgency, conf = validate_urgency_classification(
            transcript="Mild sore throat for 2 days.",
            llm_urgency="routine",
            flags=[],
        )
        assert urgency == "routine"

    def test_red_flag_tag_escalates(self):
        urgency, _ = validate_urgency_classification(
            transcript="Some symptoms.",
            llm_urgency="semi-urgent",
            flags=[{"tag": "RED_FLAG", "keyword": "fever"}],
        )
        assert urgency == "urgent"

    def test_immunocompromised_escalates_routine_to_semi_urgent(self):
        # Use risk string that appears in medical_history_str (joined and lowercased)
        urgency, conf = validate_urgency_classification(
            transcript="Mild sore throat.",
            llm_urgency="routine",
            flags=[],
            patient_history={"medicalHistory": ["cancer"]},
        )
        assert urgency == "semi-urgent"

    def test_downgrade_urgent_to_routine_when_evidence_is_routine(self):
        """LLM says urgent but transcript/summary indicate mild, improving, no red flags -> downgrade to routine."""
        urgency, conf = validate_urgency_classification(
            transcript="chief_complaint:congestion. symptom_duration:1 day. symptom_severity:2 out of 10. symptom_progression:getting better. red_flags:No. risk_factors:No",
            llm_urgency="urgent",
            flags=[{"tag": "SYMPTOM", "keyword": "congestion"}, {"tag": "SEVERITY", "keyword": "mild"}],
            summary="Patient presents with mild congestion. Symptoms improving. No red flags.",
        )
        assert urgency == "routine"
        assert conf == "high"

    def test_slot_red_flags_no_does_not_escalate(self):
        """Transcript with red_flags:No (slot format) must NOT trigger Rule 1 escalation."""
        urgency, _ = validate_urgency_classification(
            transcript="chief_complaint:sore throat. symptom_severity:mild. red_flags:No. risk_factors:No",
            llm_urgency="routine",
            flags=[],
        )
        assert urgency == "routine"

    def test_natural_language_red_flag_escalates(self):
        """Transcript with 'red flag' phrase (natural language) must escalate to urgent."""
        urgency, _ = validate_urgency_classification(
            transcript="Patient reports symptoms. Clinician noted red flag for breathing difficulty.",
            llm_urgency="routine",
            flags=[],
        )
        assert urgency == "urgent"

    def test_red_flag_in_flags_escalates(self):
        """RED_FLAG in flags list must escalate regardless of transcript."""
        urgency, _ = validate_urgency_classification(
            transcript="Mild congestion, improving.",
            llm_urgency="routine",
            flags=[{"tag": "RED_FLAG", "keyword": "fever"}],
        )
        assert urgency == "urgent"
