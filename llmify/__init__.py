from typing import TYPE_CHECKING
from .messages import (
    Message,
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolResultMessage,
    ToolCall,
    Function,
    ContentPartTextParam,
    ContentPartImageParam,
    ImageURL,
)
from .providers import (
    ChatModel,
    ChatInvokeCompletion,
    ChatInvokeUsage,
)
from .tools import (
    Tool,
    FunctionTool,
    RawSchemaTool,
    tool,
)

if TYPE_CHECKING:
    from .providers.openai import ChatOpenAI
    from .providers.azure import ChatAzureOpenAI
    from .providers.openai_compatible import OpenAICompatible
    from .providers.anthropic import ChatAnthropic


def __getattr__(name: str):
    if name == "ChatOpenAI":
        from .providers.openai import ChatOpenAI

        return ChatOpenAI

    if name == "ChatAzureOpenAI":
        from .providers.azure import ChatAzureOpenAI

        return ChatAzureOpenAI

    if name == "OpenAICompatible":
        from .providers.openai_compatible import OpenAICompatible

        return OpenAICompatible

    if name == "ChatAnthropic":
        from .providers.anthropic import ChatAnthropic

        return ChatAnthropic

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Message",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolResultMessage",
    "ToolCall",
    "Function",
    "ContentPartTextParam",
    "ContentPartImageParam",
    "ImageURL",
    "ChatOpenAI",
    "ChatAzureOpenAI",
    "ChatAnthropic",
    "ChatModel",
    "OpenAICompatible",
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    "Tool",
    "FunctionTool",
    "RawSchemaTool",
    "tool",
]
