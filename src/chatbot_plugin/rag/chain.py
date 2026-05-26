"""RAG chain — combines retrieval, prompt, and LLM.

Spec reference: specs/rag-pipeline.md — RAG Chain
"""

from chatbot_plugin.llm.resilient_llm_service import ResilientLLMService
from chatbot_plugin.rag.prompt import build_messages


async def rag_generate(
    query: str,
    articles: list[dict],
    llm_service: ResilientLLMService,
) -> str:
    """Run the RAG chain: build messages → call LLM → return reply.

    Args:
        query: User's question.
        articles: Retrieved articles with 'title' and 'content'.
        llm_service: Resilient LLM service with fallback chain.

    Returns:
        LLM-generated reply string.

    Raises:
        RuntimeError: If all LLM providers fail.
    """
    system_prompt, human_prompt = build_messages(query, articles)
    result = await llm_service.generate(system_prompt, human_prompt)
    if result is None:
        raise RuntimeError("All LLM providers failed")
    return result
