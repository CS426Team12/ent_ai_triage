from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env from the service root (ent_ai_triage/) so uvicorn works when CWD is repo parent (e.g. ai_service/).
_SERVICE_ROOT = Path(__file__).resolve().parent.parent
_DOTENV_PATH = _SERVICE_ROOT / ".env"
_ENV_FILE_TUPLE = (str(_DOTENV_PATH),) if _DOTENV_PATH.is_file() else (".env",)


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

    # Ollama (local/remote) — primary model for triage generation
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL_NAME: str = "better-triage"
    LLM_PROVIDER: str = "ollama"

    # LLM-as-a-judge (Groq)
    ENABLE_LLM_JUDGE: bool = True
    TRUST_LLM_URGENCY: bool = False  # When True, skip urgency validation and trust finetuned model
    GROQ_API_KEY: str | None = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # CORS
    ALLOWED_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE_TUPLE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
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
