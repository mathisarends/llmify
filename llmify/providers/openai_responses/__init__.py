"""OpenAI Responses provider, transports, and public result types."""

_CLIENT_EXPORTS = {"ChatOpenAIResponses", "ReasoningEffort", "ToolExecutor"}
_TRANSPORT_EXPORTS = {
    "HTTPResponsesTransport",
    "ResponsesSession",
    "ResponsesTransport",
    "WebSocketResponsesTransport",
}
_TYPE_EXPORTS = {
    "ContinuationMode",
    "OpenAIResponsesCompletion",
    "OpenAIResponsesState",
    "OpenAIResponsesStreamEnd",
    "OpenAIResponsesStreamEvent",
    "OpenAIResponsesStreamEventType",
    "OpenAIResponsesUsage",
    "PromptCacheOptions",
    "ReasoningSummary",
    "ResponsesOptions",
    "StreamOutputItemAdded",
    "StreamOutputItemDone",
    "StreamReasoningSummaryDelta",
}


def __getattr__(name: str):
    if name in _CLIENT_EXPORTS:
        from . import client

        return getattr(client, name)
    if name in _TRANSPORT_EXPORTS:
        from . import transport

        return getattr(transport, name)
    if name in _TYPE_EXPORTS:
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "ChatOpenAIResponses",
    "ContinuationMode",
    "HTTPResponsesTransport",
    "OpenAIResponsesCompletion",
    "OpenAIResponsesState",
    "OpenAIResponsesStreamEnd",
    "OpenAIResponsesStreamEvent",
    "OpenAIResponsesStreamEventType",
    "OpenAIResponsesUsage",
    "PromptCacheOptions",
    "ReasoningEffort",
    "ReasoningSummary",
    "ResponsesOptions",
    "ResponsesSession",
    "ResponsesTransport",
    "StreamOutputItemAdded",
    "StreamOutputItemDone",
    "StreamReasoningSummaryDelta",
    "ToolExecutor",
    "WebSocketResponsesTransport",
]
