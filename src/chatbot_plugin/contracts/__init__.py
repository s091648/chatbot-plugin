"""OpenAI-compatible chat completion contracts."""

from chatbot_plugin.contracts.chat_completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionMessage,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionUsage,
)

__all__ = [
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "ChatCompletionMessage",
    "ChatCompletionChoice",
    "ChatCompletionChoiceMessage",
    "ChatCompletionUsage",
]
