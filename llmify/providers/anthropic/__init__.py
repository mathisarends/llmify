"""Anthropic provider and its public result types."""

_CLIENT_EXPORTS = {"AnthropicModel", "ChatAnthropic"}
_TYPE_EXPORTS = {
    "AnthropicCompletion",
    "AnthropicStreamEnd",
    "AnthropicStreamEvent",
    "AnthropicUsage",
}


def __getattr__(name: str):
    if name in _CLIENT_EXPORTS:
        from . import client

        return getattr(client, name)
    if name in _TYPE_EXPORTS:
        from . import types

        return getattr(types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "AnthropicCompletion",
    "AnthropicModel",
    "AnthropicStreamEnd",
    "AnthropicStreamEvent",
    "AnthropicUsage",
    "ChatAnthropic",
]
