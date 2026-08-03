"""FastAPI router — single OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import json
import logging
import secrets
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from chatbot_plugin.config import CHAT_SERVICE_API_KEY
from chatbot_plugin.llm.base import AllProvidersExhausted, TextDelta, ThinkingDelta
from chatbot_plugin.contracts.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
)
from chatbot_plugin.services.chat_service import (
    ChatService,
    SourcesReady,
    StreamFailed,
    ToolCallFinished,
    ToolCallStarted,
)

logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/v1")

_STREAM_INTERRUPTED_SUFFIX = "\n\n⚠️ The response was interrupted by a provider error. Please try asking again."


def _get_chat_service(request: Request) -> ChatService:
    service: ChatService | None = getattr(request.app.state, "chat_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="ChatService not initialised")
    return service


def _check_api_key(request: Request) -> None:
    if not CHAT_SERVICE_API_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != CHAT_SERVICE_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@api_router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    """OpenAI-compatible chat completions with RAG context."""
    _check_api_key(request)
    service = _get_chat_service(request)

    last_message = req.get_last_user_message()
    if not last_message.strip():
        raise HTTPException(
            status_code=400,
            detail="messages must contain at least one user message with non-empty content",
        )

    if not req.stream:
        try:
            result = await service.chat(last_message, topic_id=req.topic_id, pinned_article_ids=req.pinned_article_ids)
        except AllProvidersExhausted:
            raise HTTPException(
                status_code=503,
                detail="All LLM providers are currently unavailable. Please try again later.",
            )
        return ChatCompletionResponse(
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionChoiceMessage(content=result.reply),
                )
            ],
        )

    # Starlette sends the HTTP status line as soon as StreamingResponse starts iterating its
    # body — before that point we can still raise HTTPException for a clean 503; after it, the
    # 200 is already committed and a total failure can only be reported inline in the stream.
    # So the first event is pulled here, outside the generator, while that's still possible.
    event_stream = service.chat_stream(last_message, topic_id=req.topic_id, pinned_article_ids=req.pinned_article_ids)
    try:
        first_event = await event_stream.__anext__()
    except StopAsyncIteration:
        first_event = None
    except AllProvidersExhausted:
        raise HTTPException(
            status_code=503,
            detail="All LLM providers are currently unavailable. Please try again later.",
        )

    cid = f"chatcmpl-{secrets.token_hex(12)}"
    ts = int(time.time())

    def _content_frame(text: str) -> bytes:
        chunk = {
            "id": cid, "object": "chat.completion.chunk",
            "created": ts, "model": req.model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        return f"data: {json.dumps(chunk)}\n\n".encode()

    def _event_to_frames(event) -> list[bytes]:
        if isinstance(event, ThinkingDelta):
            return [f"data: {json.dumps({'thinking': event.text})}\n\n".encode()]
        if isinstance(event, TextDelta):
            return [_content_frame(event.text)]
        if isinstance(event, ToolCallStarted):
            tool_call_chunk = {
                "id": cid, "object": "chat.completion.chunk",
                "created": ts, "model": req.model,
                "choices": [{
                    "index": 0,
                    "delta": {"tool_calls": [{
                        "id": event.id,
                        "type": "function",
                        "function": {"name": event.name, "arguments": json.dumps(event.arguments)},
                    }]},
                    "finish_reason": None,
                }],
            }
            return [f"data: {json.dumps(tool_call_chunk)}\n\n".encode()]
        if isinstance(event, ToolCallFinished):
            tool_result_payload = {
                "tool_result": {
                    "tool_call_id": event.id,
                    "content": event.result_summary,
                    "is_error": event.is_error,
                }
            }
            return [f"data: {json.dumps(tool_result_payload)}\n\n".encode()]
        if isinstance(event, SourcesReady):
            if not event.articles:
                return []
            sources_payload = {
                "sources": [
                    {"id": ref.id, "title": ref.title, "url": ref.url, "public_article_id": ref.public_article_id}
                    for ref in event.articles
                ]
            }
            return [f"data: {json.dumps(sources_payload)}\n\n".encode()]
        if isinstance(event, StreamFailed):
            logger.error("chat_stream_failed", extra={"error": event.message})
            return [_content_frame(_STREAM_INTERRUPTED_SUFFIX)]
        return []

    async def sse_generator():
        if first_event is not None:
            for frame in _event_to_frames(first_event):
                yield frame
        async for event in event_stream:
            for frame in _event_to_frames(event):
                yield frame

        done_chunk = {
            "id": cid, "object": "chat.completion.chunk",
            "created": ts, "model": req.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
