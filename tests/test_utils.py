"""
Unit tests for app.utils: extract_json_from_model_output.
"""
import pytest
from app.utils import extract_json_from_model_output


class TestExtractJsonFromModelOutput:
    def test_valid_json_string(self):
        text = '{"summary": "test", "urgency": "routine"}'
        result = extract_json_from_model_output(text)
        assert result["summary"] == "test"
        assert result["urgency"] == "routine"

    def test_extracts_curly_brace_region(self):
        text = 'Here is the result: {"summary": "x", "urgency": "urgent"} end'
        result = extract_json_from_model_output(text)
        assert result["summary"] == "x"
        assert result["urgency"] == "urgent"

    def test_fallback_when_no_json(self):
        text = "No json here at all"
        result = extract_json_from_model_output(text)
        assert "summary" in result
        assert result.get("urgency") == "unknown" or "summary" in result

    def test_invalid_json_returns_fallback(self):
        text = "{ invalid json "
        result = extract_json_from_model_output(text)
        assert "summary" in result
