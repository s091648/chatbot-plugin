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

FULL_PROMPT_TEMPLATE = """\
{system_prompt}

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
        snippet = CONTEXT_TEMPLATE.format(
            title=article.get("title", "Untitled"),
            content=article.get("content", ""),
        )
        if total_chars + len(snippet) > max_chars:
            # Truncate this article's content to fit
            remaining = max_chars - total_chars - len(CONTEXT_TEMPLATE.format(title=article.get("title", "Untitled"), content=""))
            if remaining > 100:
                snippet = CONTEXT_TEMPLATE.format(
                    title=article.get("title", "Untitled"),
                    content=article.get("content", "")[:remaining] + "...",
                )
                parts.append(snippet)
            break
        parts.append(snippet)
        total_chars += len(snippet)

    return "\n\n".join(parts)


def build_prompt(query: str, articles: list[dict]) -> str:
    """Build the full RAG prompt from query and articles.

    Args:
        query: User's question.
        articles: List of dicts with 'title' and 'content' keys.

    Returns:
        Complete prompt string ready for LLM.
    """
    context = build_context(articles)
    return FULL_PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context or "No relevant articles found.",
        query=query,
    )
