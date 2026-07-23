from typing import TYPE_CHECKING
from .exceptions import (
    LLMifyError,
    RetryableError,
    RateLimitError,
    OutOfCreditsError,
    ContextLengthExceededError,
    AuthenticationError,
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
from .tokens import (
    TokenTracker,
    TokenUsageEntry,
    UsageSummary,
)

if TYPE_CHECKING:
    from .providers.openai import ChatOpenAI, OpenAIModel
    from .providers.azure import ChatAzureOpenAI
    from .providers.openai_compatible import OpenAICompatible
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
    "TokenTracker",
    "TokenUsageEntry",
    "UsageSummary",
    "LLMifyError",
    "RetryableError",
    "RateLimitError",
    "OutOfCreditsError",
    "ContextLengthExceededError",
    "AuthenticationError",
]
