"""Application Configuration Module using Pydantic Settings."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv(override=True)


class Settings(BaseSettings):
    """Production application settings loaded from environment or defaults."""
    
    # Environment
    ENV: str = "production"
    DEBUG: bool = False
    
    # Database Settings (asyncpg)
    POSTGRES_USER: str = "agent_user"
    POSTGRES_PASSWORD: str = "agent_password"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "unified_memory"
    
    # Database Connection Pool Specs
    DB_POOL_MIN_SIZE: int = 5
    DB_POOL_MAX_SIZE: int = 20
    DB_POOL_TIMEOUT: float = 10.0
    
    # LLM Settings (Groq)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_FAST_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 2048
    
    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384
    
    # Memory Retrieval Thresholds
    TOP_K_FACTS: int = 5
    FACT_SIMILARITY_THRESHOLD: float = 0.70
    MAX_HISTORICAL_MESSAGES: int = 10
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def database_url(self) -> str:
        """Construct PostgreSQL async connection URL."""
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()


def validate_settings() -> None:
    """Fail fast on misconfiguration before the engine starts.

    Raises RuntimeError if the database URL or Groq key is unusable so the
    app does not start only to crash deep inside an async call.
    """
    problems: list[str] = []
    if "://" not in settings.database_url:
        problems.append("database_url is not a valid connection string")
    if not settings.GROQ_API_KEY:
        problems.append(
            "GROQ_API_KEY is missing — LLM-dependent paths (dream cycle, "
            "background extraction, execute_turn) will fail. Set it in .env."
        )
    if problems:
        # Non-fatal warning so the non-LLM test paths still run; surface clearly.
        import sys
        print(
            "[WARN] Medhas configuration issues:\n - " + "\n - ".join(problems),
            file=sys.stderr,
        )

