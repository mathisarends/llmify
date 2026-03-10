from pydantic import BaseModel

from llmify.messages import ToolCall


class ChatInvokeUsage(BaseModel):
    prompt_tokens: int
    prompt_cached_tokens: int | None = None
    prompt_cache_creation_tokens: int | None = None
    """Anthropic only: The number of tokens used to create the cache."""
    prompt_image_tokens: int | None = None
    """Google only: The number of tokens in the image."""
    completion_tokens: int
    total_tokens: int


class ChatInvokeCompletion[T](BaseModel):
    completion: T
    thinking: str | None = None
    redacted_thinking: str | None = None
    usage: ChatInvokeUsage | None = None
    stop_reason: str | None = None
    tool_calls: list[ToolCall] = []
