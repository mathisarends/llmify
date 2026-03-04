from typing import Generic, TypeVar, Union

from pydantic import BaseModel

T = TypeVar("T", bound=Union[BaseModel, str])


class ChatInvokeUsage(BaseModel):
    """
    Usage information for a chat model invocation.
    """

    prompt_tokens: int
    """The number of tokens in the prompt (this includes the cached tokens as well. When calculating the cost, subtract the cached tokens from the prompt tokens)"""

    prompt_cached_tokens: int | None
    completion_tokens: int
    total_tokens: int


class ChatInvokeCompletion(BaseModel, Generic[T]):
    completion: T
    """The completion of the response."""

    thinking: str | None = None
    redacted_thinking: str | None = None

    usage: ChatInvokeUsage | None

    stop_reason: str | None = None
