from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

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
    tool_calls: list[ToolCall] = Field(default_factory=list)


class StreamEventType(StrEnum):
    TEXT = "text"
    TOOL_CALL = "tool_call"
    END = "end"


class StreamTextDelta(BaseModel):
    type: Literal[StreamEventType.TEXT] = StreamEventType.TEXT
    delta: str


class StreamToolCall(BaseModel):
    """Emitted once a tool call's arguments JSON is fully assembled."""

    type: Literal[StreamEventType.TOOL_CALL] = StreamEventType.TOOL_CALL
    tool_call: ToolCall


class StreamEnd(BaseModel):
    """Final event. Always emitted exactly once at the end of the stream."""

    type: Literal[StreamEventType.END] = StreamEventType.END
    stop_reason: str | None = None
    usage: ChatInvokeUsage | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    completion: str = ""


type StreamEvent = StreamTextDelta | StreamToolCall | StreamEnd
