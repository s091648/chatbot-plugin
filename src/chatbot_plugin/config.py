"""Configuration for the chatbot plugin.

All settings are read from environment variables with sensible defaults.
"""

from pydantic_settings import BaseSettings


class ChatbotSettings(BaseSettings):
    """Chatbot plugin settings."""

    # LLM provider
    llm_provider: str = "claude"  # claude | gemini | openrouter
    llm_model: str = "claude-sonnet-4-6-20250514"
    llm_api_key_env: str = "ANTHROPIC_API_KEY"

    # Behavior
    max_context_articles: int = 10
    max_context_tokens: int = 8000

    model_config = {"env_prefix": "CHATBOT_"}


settings = ChatbotSettings()
