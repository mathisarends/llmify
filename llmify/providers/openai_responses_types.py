from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEnd,
    StreamProviderEvent,
    StreamTextDelta,
    StreamToolCall,
)


class ContinuationMode(StrEnum):
    STATELESS = "stateless"
    PREVIOUS_RESPONSE_ID = "previous_response_id"


type ReasoningSummary = Literal["auto", "concise", "detailed"]


class PromptCacheOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["implicit", "explicit"] = "implicit"
    ttl: Literal["30m"] | None = None


class ResponsesOptions(BaseModel):
    """Responses-only request, continuation, and transport behavior."""

    model_config = ConfigDict(extra="forbid")

    continuation_mode: ContinuationMode = ContinuationMode.STATELESS
    preserve_reasoning: bool = True
    reasoning_summary: ReasoningSummary | None = None
    prompt_cache_key: str | None = None
    prompt_cache_options: PromptCacheOptions | None = None


class OpenAIResponsesState(BaseModel):
    """Serializable state for an explicitly managed Responses conversation.

    ``input_items`` is the complete local replay window. ``output_items`` holds
    the native items from the most recent response for inspection. The former
    lets a previous-response chain safely fall back to stateless replay.
    """

    continuation_mode: ContinuationMode = ContinuationMode.STATELESS
    input_items: list[dict[str, Any]] = Field(default_factory=list)
    output_items: list[dict[str, Any]] = Field(default_factory=list)
    response_id: str | None = None
    instructions: str | None = None


class OpenAIResponsesUsage(ChatInvokeUsage):
    prompt_cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None


class OpenAIResponsesCompletion[T](ChatInvokeCompletion[T]):
    usage: OpenAIResponsesUsage | None = None
    reasoning_summary: str | None = None
    provider_state: OpenAIResponsesState


class OpenAIResponsesStreamEventType(StrEnum):
    REASONING_SUMMARY = "reasoning_summary"
    OUTPUT_ITEM_ADDED = "output_item_added"
    OUTPUT_ITEM_DONE = "output_item_done"


class StreamReasoningSummaryDelta(StreamProviderEvent):
    type: Literal[OpenAIResponsesStreamEventType.REASONING_SUMMARY] = (
        OpenAIResponsesStreamEventType.REASONING_SUMMARY
    )
    delta: str


class StreamOutputItemAdded(StreamProviderEvent):
    type: Literal[OpenAIResponsesStreamEventType.OUTPUT_ITEM_ADDED] = (
        OpenAIResponsesStreamEventType.OUTPUT_ITEM_ADDED
    )
    output_index: int
    item: dict[str, Any]


class StreamOutputItemDone(StreamProviderEvent):
    type: Literal[OpenAIResponsesStreamEventType.OUTPUT_ITEM_DONE] = (
        OpenAIResponsesStreamEventType.OUTPUT_ITEM_DONE
    )
    output_index: int
    item: dict[str, Any]


class OpenAIResponsesStreamEnd(StreamEnd):
    usage: OpenAIResponsesUsage | None = None
    reasoning_summary: str | None = None
    provider_state: OpenAIResponsesState


type OpenAIResponsesStreamEvent = (
    StreamTextDelta
    | StreamToolCall
    | StreamReasoningSummaryDelta
    | StreamOutputItemAdded
    | StreamOutputItemDone
    | OpenAIResponsesStreamEnd
)
