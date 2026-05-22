"""Configuration for the chatbot plugin.

All settings are read from environment variables with the `CHATBOT_` prefix.
"""

from pydantic_settings import BaseSettings


class ChatbotSettings(BaseSettings):
    """Chatbot plugin settings."""

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot_plugin"

    # LLM provider
    llm_provider: str = "claude"  # claude | gemini | openrouter
    llm_model: str = "claude-sonnet-4-6-20250514"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"

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
