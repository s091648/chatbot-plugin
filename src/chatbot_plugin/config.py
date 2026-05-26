"""Configuration for the chatbot plugin.

All settings are read from environment variables with the `CHATBOT_` prefix.
LLM provider configuration is loaded from providers.toml (see specs/rag-pipeline.md).
"""

from pydantic_settings import BaseSettings


class ChatbotSettings(BaseSettings):
    """Chatbot plugin settings."""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin"

    # LLM providers path (defaults to providers.toml in project root)
    llm_providers_path: str = ""

    # RAG behavior
    max_context_articles: int = 10
    max_context_tokens: int = 8000

    # Embedding (Phase 2+)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    rrf_k: int = 60
    chunk_size: int = 512
    chunk_overlap: int = 50

    model_config = {"env_prefix": "CHATBOT_"}


settings = ChatbotSettings()
