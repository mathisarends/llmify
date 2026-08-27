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
    StreamProviderEvent,
    StreamTextDelta,
    StreamToolCall,
    StreamEnd,
    StreamEvent,
)
from .retries import RetryCallback, RetryEvent
from .tools import (
    Tool,
    FunctionTool,
    RawSchemaTool,
    tool,
)

if TYPE_CHECKING:
    from .providers.openai import ChatOpenAI, OpenAIModel
    from .providers.azure import ChatAzureOpenAI, ChatAzureOpenAIResponses
    from .providers.cerebras import ChatCerebras, CerebrasModel
    from .providers.codex import (
        ChatCodex,
        CodexCliAuth,
        CodexCredentials,
        CodexCredentialsError,
    )
    from .providers.openai_compatible import OpenAICompatible
    from .providers.openai_responses import (
        ChatOpenAIResponses,
        HTTPResponsesTransport,
        ReasoningEffort,
        ResponsesSession,
        ResponsesTransport,
        WebSocketResponsesTransport,
    )
    from .providers.openai_responses.types import (
        ContinuationMode,
        OpenAIResponsesCompletion,
        OpenAIResponsesState,
        OpenAIResponsesStreamEnd,
        OpenAIResponsesStreamEventType,
        OpenAIResponsesUsage,
        PromptCacheOptions,
        ResponsesOptions,
        StreamOutputItemAdded,
        StreamOutputItemDone,
        StreamReasoningSummaryDelta,
    )
    from .providers.anthropic import (
        AnthropicModel,
        AnthropicCompletion,
        AnthropicStreamEnd,
        AnthropicUsage,
        ChatAnthropic,
    )
    from .providers.google import (
        GoogleCompletion,
        GoogleModel,
        GoogleStreamEnd,
        GoogleUsage,
        ChatGoogle,
    )


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

    if name == "ChatAzureOpenAIResponses":
        from .providers.azure import ChatAzureOpenAIResponses

        return ChatAzureOpenAIResponses

    if name == "ChatCerebras":
        from .providers.cerebras import ChatCerebras

        return ChatCerebras

    if name == "CerebrasModel":
        from .providers.cerebras import CerebrasModel

        return CerebrasModel

    if name == "ChatCodex":
        from .providers.codex import ChatCodex

        return ChatCodex

    if name == "CodexCliAuth":
        from .providers.codex import CodexCliAuth

        return CodexCliAuth

    if name == "CodexCredentials":
        from .providers.codex import CodexCredentials

        return CodexCredentials

    if name == "CodexCredentialsError":
        from .providers.codex import CodexCredentialsError

        return CodexCredentialsError

    if name == "ChatOpenAIResponses":
        from .providers.openai_responses import ChatOpenAIResponses

        return ChatOpenAIResponses

    if name == "ReasoningEffort":
        from .providers.openai_responses import ReasoningEffort

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
        from .providers import openai_responses

        return getattr(openai_responses, name)

    if name in {
        "HTTPResponsesTransport",
        "ResponsesSession",
        "ResponsesTransport",
        "WebSocketResponsesTransport",
    }:
        from .providers import openai_responses

        return getattr(openai_responses, name)

    if name == "OpenAICompatible":
        from .providers.openai_compatible import OpenAICompatible

        return OpenAICompatible

    if name == "ChatAnthropic":
        from .providers.anthropic import ChatAnthropic

        return ChatAnthropic

    if name == "AnthropicModel":
        from .providers.anthropic import AnthropicModel

        return AnthropicModel

    if name in {
        "AnthropicCompletion",
        "AnthropicStreamEnd",
        "AnthropicUsage",
    }:
        from .providers import anthropic

        return getattr(anthropic, name)

    if name == "ChatGoogle":
        from .providers.google import ChatGoogle

        return ChatGoogle

    if name == "GoogleModel":
        from .providers.google import GoogleModel

        return GoogleModel

    if name in {
        "GoogleCompletion",
        "GoogleStreamEnd",
        "GoogleUsage",
    }:
        from .providers import google

        return getattr(google, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # PEP 562: without this, the lazily exported names are invisible to dir()
    # and therefore to REPL and IDE completion.
    return sorted({*globals(), *__all__})


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
    "ChatAzureOpenAIResponses",
    "ChatCerebras",
    "CerebrasModel",
    "ChatCodex",
    "CodexCliAuth",
    "CodexCredentials",
    "CodexCredentialsError",
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
    "AnthropicModel",
    "AnthropicCompletion",
    "AnthropicStreamEnd",
    "AnthropicUsage",
    "ChatGoogle",
    "GoogleModel",
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
    "RetryCallback",
    "RetryEvent",
]
