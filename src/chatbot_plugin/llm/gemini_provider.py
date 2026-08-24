from __future__ import annotations

import asyncio
import logging

from google import genai
from opentelemetry import trace

from chatbot_plugin_sdk import RateLimitExhausted
from chatbot_plugin.llm.base import LLMResult, TextDelta, ThinkingDelta, ToolCallRequest, ToolSpec

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer(__name__)


class GeminiProvider:
    """Google Gemini LLM provider."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _build_request(messages: list[dict], max_tokens: int, tools: list[ToolSpec] | None):
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        system_instruction = "\n\n".join(system_parts) or None
        contents = [GeminiProvider._to_gemini_content(m) for m in messages if m.get("role") != "system"]

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "automatic_function_calling": genai.types.AutomaticFunctionCallingConfig(disable=True),
            # thinking_budget=-1 is Gemini's AUTOMATIC sentinel — some models (e.g.
            # gemini-2.5-flash-lite) default thinking to 0 (disabled) otherwise, so without
            # this there'd be no thought to summarize regardless of include_thoughts.
            # include_thoughts=True is what actually asks for the human-readable summary back;
            # without it the model still thinks (and thought_signature still shows up on tool
            # calls) but the summary text is withheld.
            "thinking_config": genai.types.ThinkingConfig(include_thoughts=True, thinking_budget=-1),
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if tools:
            config_kwargs["tools"] = [
                genai.types.Tool(function_declarations=[
                    genai.types.FunctionDeclaration(name=t.name, description=t.description, parameters=t.input_schema)
                    for t in tools
                ])
            ]
        return contents, genai.types.GenerateContentConfig(**config_kwargs)

    async def complete(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ) -> LLMResult:
        contents, config = self._build_request(messages, max_tokens, tools)

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                ),
            )
        except Exception as e:
            error_str = str(e)
            if "RESOURCE_EXHAUSTED" in error_str and "PerDay" in error_str:
                raise RateLimitExhausted(f"Daily quota exceeded for {self.model}") from e
            raise

        if not response.candidates:
            return LLMResult(thinking=None, text="")

        candidate = response.candidates[0]
        fr = candidate.finish_reason
        fr_name = fr.name if hasattr(fr, "name") else str(fr)
        if fr_name not in ("STOP", "1"):
            logger.warning("gemini_blocked", extra={"model": self.model, "finish_reason": fr_name})
            if fr_name != "MAX_TOKENS":
                return LLMResult(thinking=None, text="")

        content_parts = candidate.content.parts if candidate.content else []
        thinking_chunks: list[str] = []
        reply_chunks: list[str] = []
        tool_calls: list[ToolCallRequest] = []

        for i, p in enumerate(content_parts):
            fc = getattr(p, "function_call", None)
            if fc is not None:
                tool_calls.append(ToolCallRequest(
                    id=f"call_{i}",
                    name=fc.name,
                    arguments=dict(fc.args or {}),
                    thought_signature=getattr(p, "thought_signature", None),
                ))
                continue
            text = getattr(p, "text", None)
            if not text:
                continue
            if getattr(p, "thought", False):
                thinking_chunks.append(text)
            else:
                reply_chunks.append(text)

        if not reply_chunks and not thinking_chunks and not tool_calls:
            reply_chunks.append(response.text or "")

        thinking = "".join(thinking_chunks).strip() or None
        reply = "".join(reply_chunks).strip()

        if not reply and not tool_calls:
            # A normal (non-blocked) finish with no text and no tool call — the model produced
            # thinking (or nothing at all) but never committed to either an answer or a next
            # action. finish_reason is a legitimate STOP here, so the check above doesn't catch
            # this; the caller (ChatService) can't tell "no text" apart from "no text because I
            # forgot to look" without this log — see the equivalent check in stream() below.
            logger.warning(
                "gemini_no_actionable_output",
                extra={"model": self.model, "finish_reason": fr_name, "has_thinking": thinking is not None},
            )

        logger.info(
            "gemini_api_called",
            extra={
                "model": self.model,
                "finish_reason": fr_name,
                "reply_len": len(reply),
                "has_thinking": thinking is not None,
                "tool_call_count": len(tool_calls),
            },
        )
        return LLMResult(thinking=thinking, text=reply, tool_calls=tool_calls)

    async def stream(
        self,
        messages: list[dict],
        max_tokens: int,
        tools: list[ToolSpec] | None = None,
    ):
        contents, config = self._build_request(messages, max_tokens, tools)

        finish_reason_name: str | None = None
        produced_text = False
        produced_tool_call = False
        # start_span() + manual end() in a try/finally, NOT start_as_current_span() as a
        # `with` — this generator `yield`s to its caller (chat_service.py's own
        # chat.pinned_round span) while this span is still open, and per the same
        # opentelemetry-python gotcha documented on chat.llm_stream/chat.pinned_round in
        # chat_service.py, a span held open across a `yield` must not use the
        # "current span" context-manager form.
        #
        # "first_chunk" vs "first_content" are deliberately two different events: Gemini
        # can send preliminary chunks (empty candidates, or a candidate with no usable
        # part) before any real output — separating "a chunk arrived at all" from "a
        # chunk with actual text/tool-call content arrived" tells you whether a slow
        # round is waiting on the network/TTFB or on Gemini silently thinking through a
        # string of not-yet-useful chunks.
        span = _tracer.start_span("gemini.stream", attributes={"model": self.model})
        chunk_count = 0
        try:
            try:
                response_stream = await self._client.aio.models.generate_content_stream(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                span.add_event("request_sent")
                tool_call_index = 0
                first_content_seen = False
                async for chunk in response_stream:
                    if chunk_count == 0:
                        span.add_event("first_chunk")
                    chunk_count += 1
                    if not chunk.candidates:
                        continue
                    candidate = chunk.candidates[0]
                    fr = candidate.finish_reason
                    if fr is not None:
                        finish_reason_name = fr.name if hasattr(fr, "name") else str(fr)
                    # The terminal chunk carrying finish_reason often has content=None (nothing left
                    # to emit) — it must not be skipped before finish_reason above is read, or the
                    # blocked/empty-response checks below never see it.
                    if candidate.content is None:
                        continue
                    for p in candidate.content.parts:
                        fc = getattr(p, "function_call", None)
                        if fc is not None:
                            if not first_content_seen:
                                first_content_seen = True
                                span.add_event("first_content", {"kind": "tool_call"})
                            produced_tool_call = True
                            yield ToolCallRequest(
                                id=f"call_{tool_call_index}",
                                name=fc.name,
                                arguments=dict(fc.args or {}),
                                thought_signature=getattr(p, "thought_signature", None),
                            )
                            tool_call_index += 1
                            continue
                        text = getattr(p, "text", None)
                        if not text:
                            continue
                        if getattr(p, "thought", False):
                            if not first_content_seen:
                                first_content_seen = True
                                span.add_event("first_content", {"kind": "thinking"})
                            yield ThinkingDelta(text=text)
                        else:
                            if not first_content_seen:
                                first_content_seen = True
                                span.add_event("first_content", {"kind": "text"})
                            produced_text = True
                            yield TextDelta(text=text)
            except Exception as e:
                error_str = str(e)
                if "RESOURCE_EXHAUSTED" in error_str and "PerDay" in error_str:
                    raise RateLimitExhausted(f"Daily quota exceeded for {self.model}") from e
                raise
        finally:
            span.set_attribute("chunk_count", chunk_count)
            span.end()

        if finish_reason_name not in (None, "STOP", "1", "MAX_TOKENS"):
            logger.warning(
                "gemini_stream_blocked",
                extra={"model": self.model, "finish_reason": finish_reason_name},
            )
        elif not produced_text and not produced_tool_call:
            # Mirrors the equivalent check in complete() — a clean STOP that produced neither an
            # answer nor a next action (only thinking, or nothing at all). This is exactly the
            # failure mode chat_service.py's tool-call follow-up turn hit: the model's own
            # thinking said it wanted to search again, but tools=None on that turn meant it
            # couldn't — instead of falling back to answering with what it already had, it just
            # ended the turn with nothing.
            logger.warning(
                "gemini_stream_empty_response",
                extra={"model": self.model, "finish_reason": finish_reason_name},
            )

        logger.info(
            "gemini_stream_completed",
            extra={"model": self.model, "finish_reason": finish_reason_name},
        )

    @staticmethod
    def _to_gemini_content(message: dict):
        role = message["role"]
        if role == "assistant" and message.get("tool_calls"):
            parts = [
                genai.types.Part(
                    function_call=genai.types.FunctionCall(name=tc["name"], args=tc["arguments"]),
                    thought_signature=tc.get("thought_signature"),
                )
                for tc in message["tool_calls"]
            ]
            return genai.types.Content(role="model", parts=parts)
        if role == "tool":
            payload = {"error": message["content"]} if message.get("is_error") else {"result": message["content"]}
            return genai.types.Content(
                role="user",
                parts=[genai.types.Part(function_response=genai.types.FunctionResponse(
                    name=message["name"], response=payload,
                ))],
            )
        gemini_role = "model" if role == "assistant" else "user"
        return genai.types.Content(role=gemini_role, parts=[genai.types.Part(text=message["content"])])
