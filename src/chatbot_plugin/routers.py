"""FastAPI router — single OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from chatbot_plugin.contracts.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
)
from chatbot_plugin_sdk import RagQueryProcessor

api_router = APIRouter(prefix="/v1")


def _get_processor(request: Request) -> RagQueryProcessor:
    processor: RagQueryProcessor | None = \
        getattr(request.app.state, "processor", None)
    if processor is None:
        raise HTTPException(status_code=500, detail="RagQueryProcessor not initialised")
    return processor


@api_router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completions with RAG context."""
    processor = _get_processor(request)

    last_message = req.get_last_user_message()
    if not last_message.strip():
        raise HTTPException(
            status_code=400,
            detail="messages must contain at least one user message with non-empty content",
        )

    result = await processor.chat(last_message)

    return ChatCompletionResponse(
        model="rag-default",
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=result.reply),
            )
        ],
    )
