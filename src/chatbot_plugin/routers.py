"""FastAPI router — single OpenAI-compatible /v1/chat/completions endpoint."""

from __future__ import annotations

import json
import os
import secrets
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from chatbot_plugin.contracts.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
)
from chatbot_plugin.chat_service import ChatService

api_router = APIRouter(prefix="/v1")

_API_KEY = os.getenv("CHAT_SERVICE_API_KEY", "")


def _get_chat_service(request: Request) -> ChatService:
    service: ChatService | None = \
        getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="ChatService not initialised")
    return service


def _check_api_key(request: Request) -> None:
    if not _API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@api_router.post("/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
):
    """OpenAI-compatible chat completions with RAG context."""
    _check_api_key(request)
    service = _get_chat_service(request)

    last_message = req.get_last_user_message()
    if not last_message.strip():
        raise HTTPException(
            status_code=400,
            detail="messages must contain at least one user message with non-empty content",
        )

    result = await service.chat(last_message, topic_id=req.topic_id)

    if req.stream:
        cid = f"chatcmpl-{secrets.token_hex(12)}"
        ts = int(time.time())

        async def sse_generator():
            content_chunk = {
                "id": cid, "object": "chat.completion.chunk",
                "created": ts, "model": req.model,
                "choices": [{"index": 0, "delta": {"content": result.reply}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n".encode()

            done_chunk = {
                "id": cid, "object": "chat.completion.chunk",
                "created": ts, "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_chunk)}\n\n".encode()

            if result.articles_used:
                sources_payload = {
                    "sources": [
                        {
                            "id": ref.id,
                            "title": ref.title,
                            "url": ref.url,
                            "public_article_id": ref.public_article_id,
                        }
                        for ref in result.articles_used
                    ]
                }
                yield f"data: {json.dumps(sources_payload)}\n\n".encode()

            yield b"data: [DONE]\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionChoiceMessage(content=result.reply),
            )
        ],
    )
