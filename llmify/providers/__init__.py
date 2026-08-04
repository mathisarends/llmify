from llmify.base import ChatModel
from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEventType,
    StreamProviderEvent,
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

    if name == "ChatAzureOpenAIResponses":
        from .azure import ChatAzureOpenAIResponses

        return ChatAzureOpenAIResponses

    if name == "ChatCerebras":
        from .cerebras import ChatCerebras

        return ChatCerebras

    if name == "ChatCodex":
        from .codex import ChatCodex

        return ChatCodex

    if name == "ChatOpenAIResponses":
        from .openai_responses import ChatOpenAIResponses

        return ChatOpenAIResponses

    if name == "ReasoningEffort":
        from .openai_responses import ReasoningEffort

        return ReasoningEffort

    if name in {
        "ContinuationMode",
        "OpenAIResponsesCompletion",
        "OpenAIResponsesState",
        "OpenAIResponsesStreamEnd",
        "OpenAIResponsesStreamEventType",
        "OpenAIResponsesUsage",
        "PromptCacheOptions",
        "ResponsesOptions",
        "StreamOutputItemAdded",
        "StreamOutputItemDone",
        "StreamReasoningSummaryDelta",
    }:
        from . import openai_responses_types

        return getattr(openai_responses_types, name)

    if name in {
        "HTTPResponsesTransport",
        "ResponsesSession",
        "ResponsesTransport",
        "WebSocketResponsesTransport",
    }:
        from . import openai_responses_transport

        return getattr(openai_responses_transport, name)

    if name == "OpenAICompatible":
        from .openai_compatible import OpenAICompatible

        return OpenAICompatible

    if name == "ChatAnthropic":
        from .anthropic import ChatAnthropic

        return ChatAnthropic

    if name in {
        "AnthropicCompletion",
        "AnthropicStreamEnd",
        "AnthropicUsage",
    }:
        from . import anthropic_types

        return getattr(anthropic_types, name)

    if name == "ChatGoogle":
        from .google import ChatGoogle

        return ChatGoogle

    if name in {
        "GoogleCompletion",
        "GoogleStreamEnd",
        "GoogleUsage",
    }:
        from . import google_types

        return getattr(google_types, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatAzureOpenAIResponses",
    "ChatCerebras",
    "ChatCodex",
    "ChatOpenAIResponses",
    "ReasoningEffort",
    "ContinuationMode",
    "OpenAIResponsesCompletion",
    "OpenAIResponsesState",
    "OpenAIResponsesStreamEnd",
    "OpenAIResponsesStreamEventType",
    "OpenAIResponsesUsage",
    "PromptCacheOptions",
    "ResponsesOptions",
    "StreamOutputItemAdded",
    "StreamOutputItemDone",
    "StreamReasoningSummaryDelta",
    "HTTPResponsesTransport",
    "ResponsesSession",
    "ResponsesTransport",
    "WebSocketResponsesTransport",
    "ChatAnthropic",
    "AnthropicCompletion",
    "AnthropicStreamEnd",
    "AnthropicUsage",
    "ChatGoogle",
    "GoogleCompletion",
    "GoogleStreamEnd",
    "GoogleUsage",
    "ChatModel",
    "OpenAICompatible",
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    "StreamEventType",
    "StreamProviderEvent",
    "StreamTextDelta",
    "StreamToolCall",
    "StreamEnd",
    "StreamEvent",
]
