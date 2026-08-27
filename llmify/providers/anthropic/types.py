from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEnd,
    StreamTextDelta,
    StreamToolCall,
)


class AnthropicUsage(ChatInvokeUsage):
    prompt_cache_creation_tokens: int | None = None
    """Tokens written to the prompt cache by this request."""


class AnthropicCompletion[T](ChatInvokeCompletion[T]):
    usage: AnthropicUsage | None = None


class AnthropicStreamEnd(StreamEnd):
    usage: AnthropicUsage | None = None


type AnthropicStreamEvent = StreamTextDelta | StreamToolCall | AnthropicStreamEnd
