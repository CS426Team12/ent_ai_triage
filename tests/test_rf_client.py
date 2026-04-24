from app.rf_client import predict_rf_urgency


def test_predict_rf_urgency_returns_expected_shape():
    transcript = "[Patient]: mild sore throat for 2 days, improving."
    result = predict_rf_urgency(transcript)
    assert "urgency" in result
    assert result["urgency"] in {"routine", "semi-urgent", "urgent"}
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
    assert "source" in result
