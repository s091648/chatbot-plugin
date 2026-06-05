"""Configuration for the toolbox.

All settings are read from environment variables with the `CHATBOT_` prefix.
"""

from pydantic_settings import BaseSettings


class ChatbotSettings(BaseSettings):
    """Toolbox settings."""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin"

    # Embedding model config (must match what scrape-and-analyze uses)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimension: int = 1024
    sparse_dimension: int = 250002

    # Search / RRF
    rrf_k: int = 60
    search_candidates: int = 50
    max_context_chunks: int = 10

    # LLM (optional — only needed for /tools/chat)
    llm_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6-20250514"

    model_config = {"env_prefix": "CHATBOT_"}


settings = ChatbotSettings()
