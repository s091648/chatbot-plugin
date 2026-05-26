"""Prompt templates and context assembly for RAG.

Spec reference: specs/rag-pipeline.md — Prompt Builder
"""

from chatbot_plugin.config import settings

SYSTEM_PROMPT = """\
You are an assistant that answers questions based on the provided article excerpts.
Always cite your sources using [source: article_title] notation.
If the provided articles do not contain enough information, say so honestly.
Answer in the same language as the user's question."""

CONTEXT_TEMPLATE = """\
[source: {title}]
{content}"""

HUMAN_PROMPT_TEMPLATE = """\
## Relevant Articles

{context}

## User Question

{query}"""


def build_context(articles: list[dict], max_tokens: int | None = None) -> str:
    """Assemble article excerpts into a context string with token budget.

    Args:
        articles: List of dicts with 'title' and 'content' keys.
        max_tokens: Token budget (approx 4 chars/token). Defaults to settings.

    Returns:
        Formatted context string, truncated to budget.
    """
    if max_tokens is None:
        max_tokens = settings.max_context_tokens

    max_chars = max_tokens * 4  # rough char-to-token estimate
    parts: list[str] = []
    total_chars = 0

    for article in articles:
        title = article.get("title", "Untitled")
        content = article.get("content", "")

        # Calculate template overhead (everything except the content)
        template_overhead = len(CONTEXT_TEMPLATE.format(title=title, content=""))
        snippet_budget = max_chars - total_chars - template_overhead

        if snippet_budget <= 0:
            break

        if len(content) <= snippet_budget:
            snippet = CONTEXT_TEMPLATE.format(title=title, content=content)
        elif snippet_budget - 3 > 100:  # 3 chars for "..."
            snippet = CONTEXT_TEMPLATE.format(
                title=title,
                content=content[:snippet_budget - 3] + "...",
            )
        else:
            break

        parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n".join(parts)


def build_messages(query: str, articles: list[dict]) -> tuple[str, str]:
    """Build RAG prompt as (system_prompt, human_prompt) tuple.

    Separating system and human prompts allows LLM providers to use
    proper system message handling for better instruction following.

    Args:
        query: User's question.
        articles: List of dicts with 'title' and 'content' keys.

    Returns:
        (system_prompt, human_prompt) tuple.
    """
    context = build_context(articles)
    human_prompt = HUMAN_PROMPT_TEMPLATE.format(
        context=context or "No relevant articles found.",
        query=query,
    )
    return SYSTEM_PROMPT, human_prompt
