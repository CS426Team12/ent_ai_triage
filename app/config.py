from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration for the AI service."""

    # Database configuration (only needed if AI service writes directly to DB)
    DB_USER: str | None = None
    DB_PW: str | None = None
    DB_HOST: str | None = None
    DB_PORT: str = "5432"
    DB_NAME: str | None = None

    BACKEND_BASE_URL: str
    BACKEND_USERNAME: str
    BACKEND_PASSWORD: str


    # Redis (AI should use DB 1)
    AI_REDIS_URL: str = "redis://localhost:6379/1"

    # Ollama EC2
    OLLAMA_BASE_URL: str
    OLLAMA_MODEL_NAME: str = "phi3"

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
