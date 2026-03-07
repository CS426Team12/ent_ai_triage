from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the AI service."""

    # Database configuration
    DB_USER: str | None = None
    DB_PW: str | None = None
    DB_HOST: str | None = None
    DB_PORT: str = "5432"
    DB_NAME: str | None = None

    BACKEND_BASE_URL: str = "http://localhost:8000"
    # Service account for local auth (env vars override)
    BACKEND_USERNAME: str = "ai@test.com"
    BACKEND_PASSWORD: str = "password67"


    # Redis (AI should use DB 1)
    AI_REDIS_URL: str = "redis://localhost:6379/1"

    # Ollama (local/remote)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "phi3"

    # Hosted LLM (Groq)
    LLM_PROVIDER: str = "groq"  # "groq" or "ollama"
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True
    )

    @property
    def SQLALCHEMY_DATABASE_URL(self) -> str:
        """Optional DB URL builder."""
        if not all([self.DB_USER, self.DB_PW, self.DB_HOST, self.DB_NAME]):
            return ""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PW}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


settings = Settings()


@lru_cache()
def get_settings() -> Settings:
    return settings
