from llmify.base import ChatModel
from llmify.views import ChatInvokeCompletion, ChatInvokeUsage


def __getattr__(name: str):
    if name == "ChatOpenAI":
        from .openai import ChatOpenAI

        return ChatOpenAI

    if name == "ChatAzureOpenAI":
        from .azure import ChatAzureOpenAI

        return ChatAzureOpenAI

    if name == "OpenAICompatible":
        from .openai_compatible import OpenAICompatible

        return OpenAICompatible

    if name == "ChatAnthropic":
        from .anthropic import ChatAnthropic

        return ChatAnthropic

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatAnthropic",
    "ChatModel",
    "OpenAICompatible",
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
]
