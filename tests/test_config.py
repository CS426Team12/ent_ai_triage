"""
Unit tests for app.config: Settings and get_settings.
"""
import pytest


class TestSettings:
    """Tests for Settings class."""

    def test_get_settings_returns_settings(self):
        from app.config import get_settings
        s = get_settings()
        assert s is not None
        assert hasattr(s, "BACKEND_BASE_URL")
        assert hasattr(s, "OLLAMA_BASE_URL")
        assert hasattr(s, "OLLAMA_MODEL_NAME")
        assert hasattr(s, "LLM_PROVIDER")
        assert hasattr(s, "ENABLE_LLM_JUDGE")

    def test_allowed_origins_default(self):
        from app.config import settings
        assert settings.ALLOWED_ORIGINS is not None
        assert isinstance(settings.ALLOWED_ORIGINS, list)

    def test_sqlalchemy_database_url_empty_when_incomplete(self):
        from app.config import Settings
        # With no DB_* set, SQLALCHEMY_DATABASE_URL is ""
        s = Settings(
            BACKEND_BASE_URL="http://x",
            BACKEND_USERNAME="u",
            BACKEND_PASSWORD="p",
            OLLAMA_BASE_URL="http://y",
        )
        assert s.SQLALCHEMY_DATABASE_URL == ""

    def test_sqlalchemy_database_url_built_when_complete(self):
        from app.config import Settings
        s = Settings(
            BACKEND_BASE_URL="http://x",
            BACKEND_USERNAME="u",
            BACKEND_PASSWORD="p",
            OLLAMA_BASE_URL="http://y",
            DB_USER="u",
            DB_PW="p",
            DB_HOST="h",
            DB_NAME="n",
        )
        url = s.SQLALCHEMY_DATABASE_URL
        assert "postgresql://" in url
        assert "u:p@" in url
        assert "h" in url
        assert "n" in url
