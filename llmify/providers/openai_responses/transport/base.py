"""Shared transport protocols."""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI


@runtime_checkable
class ResponsesSession(Protocol):
    """One transport-scoped Responses conversation session."""

    def events(self, request: dict[str, Any]) -> AsyncIterator[Any]: ...

    def can_continue_from(self, response_id: str) -> bool: ...

    def remember(self, response_id: str) -> None: ...


@runtime_checkable
class ResponsesTransport(Protocol):
    """Port for opening a Responses session."""

    def session(
        self, client: AsyncOpenAI
    ) -> AbstractAsyncContextManager[ResponsesSession]: ...
