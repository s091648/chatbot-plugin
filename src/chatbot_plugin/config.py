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

    model_config = {"env_prefix": "CHATBOT_"}


settings = ChatbotSettings()
