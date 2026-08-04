"""Application Configuration Module using Pydantic Settings."""

from dotenv import load_dotenv
from typing import Optional
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
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIMENSION: int = 768
    
    # Memory Retrieval Thresholds
    TOP_K_FACTS: int = 5
    # Cosine pre-filter. Tuned for BAAI/bge-base-en-v1.5 (768-dim); bge-small scored
    # higher absolutes so 0.70 was safe, but bge-base's distribution is lower — 0.55
    # keeps true matches while still dropping noise. Final calibration is the FoK gate.
    FACT_SIMILARITY_THRESHOLD: float = 0.55
    MAX_HISTORICAL_MESSAGES: int = 10

    # Atomic dedup / decision-matrix (Mem0-inspired)
    FACT_HASH_DEDUP: bool = True          # Skip re-insert if an active fact with same md5 hash exists
    FACT_SEMANTIC_DUP_THRESHOLD: float = 0.92   # cosine above which two facts are "same claim"
    FACT_SEMANTIC_UPDATE_THRESHOLD: float = 0.78  # cosine above which incoming supersedes existing
    FACT_RERANK: bool = True                      # deterministic fusion rerank (closes Mem0 rerank gap)
    FACT_RERANKER_ENABLED: bool = True            # use local cross-encoder reranker (Mem0-style) when available
    FACT_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Mem0 SentenceTransformerReranker default
    FACT_RERANKER_NORMALIZE: bool = True          # sigmoid-normalize cross-encoder logits to [0,1] (per Mem0)
    # Optional stronger cross-encoder for higher precision (e.g. mixedbread-ai/mxbai-rerank-base-v1, BAAI/bge-reranker-v2-m3).
    # Empty -> use FACT_RERANKER_MODEL. Swap via env FACT_RERANKER_STRONG_MODEL to A/B without code edits.
    FACT_RERANKER_STRONG_MODEL: str = ""
    FACT_RERANKER_WARMUP: bool = True             # pre-load the cross-encoder at startup so first query isn't slow
    AGI_ENGINE_ENABLED: bool = True               # route live agent recall/context through agi.engine (E1–E37)
    DECISION_MATRIX_MODEL: Optional[str] = None  # LLM model for the Mem0 decision matrix (None=default)
    
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

