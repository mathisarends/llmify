from llmify.base import ChatModel
from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEventType,
    StreamTextDelta,
    StreamToolCall,
    StreamEnd,
    StreamEvent,
)


def __getattr__(name: str):
    if name == "ChatOpenAI":
        from .openai import ChatOpenAI

        return ChatOpenAI

    if name == "ChatAzureOpenAI":
        from .azure import ChatAzureOpenAI

        return ChatAzureOpenAI

    if name == "ChatCerebras":
        from .cerebras import ChatCerebras

        return ChatCerebras

    if name == "OpenAICompatible":
        from .openai_compatible import OpenAICompatible

        return OpenAICompatible

    if name == "ChatAnthropic":
        from .anthropic import ChatAnthropic

        return ChatAnthropic

    if name == "ChatGoogle":
        from .google import ChatGoogle

        return ChatGoogle

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatCerebras",
    "ChatAnthropic",
    "ChatGoogle",
    "ChatModel",
    "OpenAICompatible",
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    "StreamEventType",
    "StreamTextDelta",
    "StreamToolCall",
    "StreamEnd",
    "StreamEvent",
]
