"""Google provider and its public result types."""

_CLIENT_EXPORTS = {"ChatGoogle", "GoogleModel"}
_TYPE_EXPORTS = {
    "GoogleCompletion",
    "GoogleStreamEnd",
    "GoogleStreamEvent",
    "GoogleUsage",
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
    "ChatGoogle",
    "GoogleCompletion",
    "GoogleModel",
    "GoogleStreamEnd",
    "GoogleStreamEvent",
    "GoogleUsage",
]
