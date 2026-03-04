"""
Unit tests for app.ml_client: extract_features_from_transcript,
extract_duration, extract_pain_severity, extract_age,
extract_nasal_discharge_type, extract_symptom_keywords, predict_urgency.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.ml_client import (
    extract_features_from_transcript,
    extract_duration,
    extract_pain_severity,
    extract_age,
    extract_nasal_discharge_type,
    extract_symptom_keywords,
    predict_urgency,
)


class TestExtractDuration:
    def test_days_pattern(self):
        assert extract_duration("symptoms for 3 days") == 3
        assert extract_duration("1 day") == 1

    def test_weeks_pattern(self):
        assert extract_duration("2 weeks") == 14
        assert extract_duration("1 week") == 7

    def test_default(self):
        assert extract_duration("no duration mentioned") == 2


class TestExtractPainSeverity:
    def test_out_of_10(self):
        assert extract_pain_severity("pain 5 out of 10") == 5
        assert extract_pain_severity("7/10 pain") == 7

    def test_severe_keywords(self):
        assert extract_pain_severity("severe pain") == 8

    def test_mild_keywords(self):
        assert extract_pain_severity("mild discomfort") == 3

    def test_moderate_keywords(self):
        assert extract_pain_severity("moderate pain") == 5

    def test_default(self):
        assert extract_pain_severity("some pain") == 3


class TestExtractAge:
    def test_year_pattern(self):
        assert extract_age("patient is 45 years old") == 45
        assert extract_age("25 yo") == 25

    def test_default(self):
        assert extract_age("no age") == 40


class TestExtractNasalDischargeType:
    def test_clear(self):
        assert extract_nasal_discharge_type("clear discharge") == "clear"

    def test_yellow(self):
        assert extract_nasal_discharge_type("yellow mucus") == "yellow"

    def test_bloody(self):
        assert extract_nasal_discharge_type("bloody nose") == "bloody"

    def test_default(self):
        assert extract_nasal_discharge_type("congestion") == "clear"


class TestExtractSymptomKeywords:
    def test_finds_keywords(self):
        result = extract_symptom_keywords("sore throat and congestion")
        assert "throat" in result or "congestion" in result

    def test_general_fallback(self):
        result = extract_symptom_keywords("something random")
        assert result == "general_ent_concern" or "general" in result


class TestExtractFeaturesFromTranscript:
    def test_returns_dict_with_expected_keys(self):
        result = extract_features_from_transcript("Mild sore throat for 3 days.")
        assert "duration_days" in result
        assert "pain_severity" in result
        assert "age" in result
        assert "worsening" in result
        assert "fever" in result
        assert "symptom_keywords" in result

    def test_worsening_detected(self):
        result = extract_features_from_transcript("symptoms getting worse")
        assert result["worsening"] == 1

    def test_fever_detected(self):
        result = extract_features_from_transcript("has fever")
        assert result["fever"] == 1


class TestPredictUrgency:
    def test_returns_dict_when_no_model(self):
        with patch("app.ml_client.load_model", return_value=None):
            result = predict_urgency("sore throat")
            assert result["urgency"] == "routine"
            assert result["confidence"] == 0.0

    def test_returns_dict_when_model_incomplete(self):
        with patch("app.ml_client.load_model", return_value={"model": None}):
            result = predict_urgency("sore throat")
            assert result["urgency"] == "routine"
            assert result["confidence"] == 0.0
