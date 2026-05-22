"""RAG chain — combines retrieval, prompt, and LLM.

Spec reference: specs/rag-pipeline.md — RAG Chain
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot_plugin.config import settings
from chatbot_plugin.rag.prompt import build_prompt


def _get_llm() -> BaseChatModel:
    """Instantiate the LLM based on config."""
    provider = settings.llm_provider.lower()
    if provider == "claude":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key_env_var=settings.llm_api_key_env,
            max_tokens=2048,
        )
    elif provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key_env_var=settings.llm_api_key_env,
        )
    else:
        msg = f"Unsupported LLM provider: {provider}"
        raise ValueError(msg)


async def rag_generate(query: str, articles: list[dict]) -> str:
    """Run the RAG chain: build prompt → call LLM → return reply.

    Args:
        query: User's question.
        articles: Retrieved articles with 'title' and 'content'.

    Returns:
        LLM-generated reply string.

    Raises:
        ValueError: If LLM provider is unsupported.
    """
    llm = _get_llm()
    prompt = build_prompt(query, articles)
    response = await llm.ainvoke(prompt)
    return response.content
