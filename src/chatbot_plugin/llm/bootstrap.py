"""Bootstrap factory for building the LLM service from providers.toml."""

import os

import structlog

from chatbot_plugin.llm.config import load_providers
from chatbot_plugin.llm.rate_limit import SlidingWindowStrategy, NoOpStrategy
from chatbot_plugin.llm.resilient_llm_service import ProviderHandler, ResilientLLMService

logger = structlog.get_logger()


def build_llm_service(path: str | None = None) -> ResilientLLMService:
    """Build the resilient LLM service from providers.toml.

    Args:
        path: Path to providers.toml. Defaults to project root.

    Returns:
        Configured ResilientLLMService.

    Raises:
        ValueError: If no valid providers are configured.
        FileNotFoundError: If providers.toml does not exist.
    """
    providers = load_providers(path)
    handlers: list[ProviderHandler] = []

    for cfg in providers:
        name = cfg["name"]
        model = cfg["model"]
        api_key = os.environ.get(cfg.get("api_key_env", ""), "")

        if not api_key:
            logger.warning("provider_missing_api_key", name=name, env_var=cfg.get("api_key_env"))
            continue

        provider = _create_provider(name, api_key, model)
        if provider is None:
            logger.warning("provider_unknown", name=name)
            continue

        strategy = _create_strategy(cfg.get("strategy", {}))

        handlers.append(ProviderHandler(
            provider=provider,
            strategy=strategy,
            priority=cfg.get("priority", 999),
            name=name,
        ))

    if not handlers:
        raise ValueError("No valid LLM providers configured. Check providers.toml and API key env vars.")

    logger.info("llm_service_initialized", provider_count=len(handlers))
    return ResilientLLMService(handlers=handlers)


def _create_provider(name: str, api_key: str, model: str):
    """Instantiate a provider by name. Returns None if name is unknown."""
    if name == "claude":
        from chatbot_plugin.llm.claude_provider import ClaudeProvider
        return ClaudeProvider(api_key=api_key, model=model)
    elif name == "gemini":
        from chatbot_plugin.llm.gemini_provider import GeminiProvider
        return GeminiProvider(api_key=api_key, model=model)
    elif name == "openrouter":
        from chatbot_plugin.llm.openrouter_provider import OpenRouterProvider
        return OpenRouterProvider(api_key=api_key, model=model)
    return None


def _create_strategy(cfg: dict):
    """Create a rate limiting strategy from config dict."""
    strategy_type = cfg.get("type", "")
    if strategy_type == "sliding_window":
        return SlidingWindowStrategy(
            rpm=cfg.get("rpm", 10),
            tpm=cfg.get("tpm", 100000),
            rpd=cfg.get("rpd", 1000),
        )
    return NoOpStrategy()
