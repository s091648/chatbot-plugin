"""OpenAI-compatible chat completion contracts."""

from pydantic import BaseModel, Field


class ChatCompletionMessage(BaseModel):
    """A message in the messages array."""

    role: str = Field(..., description="one of system, user, assistant")
    content: str = Field(..., description="message text")


class ChatCompletionRequest(BaseModel):
    """POST /v1/chat/completions request body."""

    model: str = Field(default="rag-default", description="model identifier (unused, forwarded for compatibility)")
    messages: list[ChatCompletionMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    stream: bool = Field(default=False)

    def get_last_user_message(self) -> str:
        """Return the last user message, or raise if missing."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return ""


class ChatCompletionChoiceMessage(BaseModel):
    """The message object inside a choice."""

    role: str = "assistant"
    content: str = Field(...)


class ChatCompletionChoice(BaseModel):
    """A single choice in the response."""

    index: int = 0
    message: ChatCompletionChoiceMessage = Field(...)
    finish_reason: str = "stop"


class ChatCompletionUsage(BaseModel):
    """Token usage — always zero since we don't count tokens."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """POST /v1/chat/completions response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{__import__('secrets').token_hex(12)}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(__import__('time').time()))
    model: str = "rag-default"
    choices: list[ChatCompletionChoice]
    usage: ChatCompletionUsage = Field(default_factory=ChatCompletionUsage)
