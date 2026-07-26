from typing import TYPE_CHECKING
from .exceptions import (
    LLMifyError,
    RetryableError,
    RateLimitError,
    OutOfCreditsError,
    ContextLengthExceededError,
    AuthenticationError,
    CredentialsUnavailableError,
)
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
    StreamEventType,
    StreamTextDelta,
    StreamToolCall,
    StreamEnd,
    StreamEvent,
)
from .tools import (
    Tool,
    FunctionTool,
    RawSchemaTool,
    tool,
)

if TYPE_CHECKING:
    from .providers.openai import ChatOpenAI, OpenAIModel
    from .providers.azure import ChatAzureOpenAI
    from .providers.cerebras import ChatCerebras, CerebrasModel
    from .providers.codex import ChatCodex
    from .providers.openai_compatible import OpenAICompatible
    from .providers.openai_responses import ChatOpenAIResponses
    from .providers.anthropic import ChatAnthropic, AnthropicModel
    from .providers.google import ChatGoogle, GoogleModel


def __getattr__(name: str):
    if name == "ChatOpenAI":
        from .providers.openai import ChatOpenAI

        return ChatOpenAI

    if name == "OpenAIModel":
        from .providers.openai import OpenAIModel

        return OpenAIModel

    if name == "ChatAzureOpenAI":
        from .providers.azure import ChatAzureOpenAI

        return ChatAzureOpenAI

    if name == "ChatCerebras":
        from .providers.cerebras import ChatCerebras

        return ChatCerebras

    if name == "CerebrasModel":
        from .providers.cerebras import CerebrasModel

        return CerebrasModel

    if name == "ChatCodex":
        from .providers.codex import ChatCodex

        return ChatCodex

    if name == "ChatOpenAIResponses":
        from .providers.openai_responses import ChatOpenAIResponses

        return ChatOpenAIResponses

    if name == "OpenAICompatible":
        from .providers.openai_compatible import OpenAICompatible

        return OpenAICompatible

    if name == "ChatAnthropic":
        from .providers.anthropic import ChatAnthropic

        return ChatAnthropic

    if name == "AnthropicModel":
        from .providers.anthropic import AnthropicModel

        return AnthropicModel

    if name == "ChatGoogle":
        from .providers.google import ChatGoogle

        return ChatGoogle

    if name == "GoogleModel":
        from .providers.google import GoogleModel

        return GoogleModel

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
    "OpenAIModel",
    "ChatAzureOpenAI",
    "ChatCerebras",
    "CerebrasModel",
    "ChatCodex",
    "ChatOpenAIResponses",
    "ChatAnthropic",
    "AnthropicModel",
    "ChatGoogle",
    "GoogleModel",
    "ChatModel",
    "OpenAICompatible",
    "ChatInvokeCompletion",
    "ChatInvokeUsage",
    "StreamEventType",
    "StreamTextDelta",
    "StreamToolCall",
    "StreamEnd",
    "StreamEvent",
    "Tool",
    "FunctionTool",
    "RawSchemaTool",
    "tool",
    "LLMifyError",
    "RetryableError",
    "RateLimitError",
    "OutOfCreditsError",
    "ContextLengthExceededError",
    "AuthenticationError",
    "CredentialsUnavailableError",
]
