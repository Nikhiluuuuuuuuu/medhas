"""Domain-specific exception hierarchy for Unified Memory Engine."""

class MemoryEngineException(Exception):
    """Base exception for memory engine errors."""

class DatabaseConnectionError(MemoryEngineException):
    """Raised when database connection pool initialization fails."""

class StorageOperationError(MemoryEngineException):
    """Raised when a storage CRUD operation fails."""

class LLMProviderError(MemoryEngineException):
    """Raised when Groq or LLM inference fails."""

class EmbeddingGenerationError(MemoryEngineException):
    """Raised when vector embedding generation fails."""

class MemoryBlockNotFoundError(MemoryEngineException):
    """Raised when a working memory block is requested but does not exist."""
