from collections.abc import AsyncIterator
from typing import Any, Protocol, TypeVar, overload, runtime_checkable

from pydantic import BaseModel

from llmify.messages import Message, ModelResponse
from llmify.views import ChatInvokeCompletion


T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class BaseChatModel(Protocol):
    @overload
    async def invoke(
        self, messages: list[Message], output_format: None = None, **kwargs: Any
    ) -> ChatInvokeCompletion[str]: ...

    @overload
    async def invoke(
        self, messages: list[Message], output_format: type[T], **kwargs: Any
    ) -> ChatInvokeCompletion[T]: ...

    async def invoke(
        self,
        messages: list[Message],
        output_format: type[T] | None = None,
        **kwargs: Any,
    ) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]: ...

    async def invoke_tools(
        self,
        messages: list[Message],
        tools: list[type[BaseModel]],
        **kwargs: Any,
    ) -> ModelResponse: ...

    async def stream(
        self,
        messages: list[Message],
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...
