"""Persistent and prewarmable WebSocket transport for the Responses API."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from openai import AsyncOpenAI, OpenAIError
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseErrorEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
)

from llmify.exceptions import LLMifyError

from .base import ResponsesSession


class WebSocketResponsesTransport:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._prewarmed_client: AsyncOpenAI | None = None
        self._prewarmed_manager: Any | None = None
        self._prewarmed_session: _WebSocketResponsesSession | None = None

    @property
    def is_prewarmed(self) -> bool:
        """Whether this transport currently owns an open prewarmed socket."""
        return self._prewarmed_session is not None

    async def prewarm(self, client: AsyncOpenAI) -> None:
        """Open and retain a WebSocket without sending a Responses request."""
        async with self._lock:
            if self._prewarmed_session is not None:
                if self._prewarmed_client is not client:
                    raise LLMifyError(
                        "A prewarmed WebSocket transport cannot be shared between "
                        "different clients."
                    )
                return

            manager = _connection_manager(client)
            try:
                connection = await manager.__aenter__()
            except OpenAIError as exc:
                _raise_websocket_dependency_error(exc)
                raise

            self._prewarmed_client = client
            self._prewarmed_manager = manager
            self._prewarmed_session = _WebSocketResponsesSession(connection)

    async def aclose(self) -> None:
        """Close the retained prewarmed socket, if one exists."""
        async with self._lock:
            await self._close_prewarmed()

    @asynccontextmanager
    async def session(
        self, client: AsyncOpenAI
    ) -> AsyncGenerator[ResponsesSession, None]:
        async with self._lock:
            if self._prewarmed_session is not None:
                if self._prewarmed_client is not client:
                    raise LLMifyError(
                        "A prewarmed WebSocket transport cannot be shared between "
                        "different clients."
                    )
                try:
                    yield self._prewarmed_session
                except BaseException as exc:
                    # A cancelled/partial response leaves unread events on this
                    # channel, so it must not be handed to the next request.
                    await self._close_prewarmed(exc)
                    raise
                return

            manager = _connection_manager(client)
            try:
                async with manager as connection:
                    yield _WebSocketResponsesSession(connection)
            except OpenAIError as exc:
                _raise_websocket_dependency_error(exc)
                raise

    async def _close_prewarmed(self, exc: BaseException | None = None) -> None:
        manager = self._prewarmed_manager
        self._prewarmed_client = None
        self._prewarmed_manager = None
        self._prewarmed_session = None
        if manager is not None:
            await manager.__aexit__(
                type(exc) if exc is not None else None,
                exc,
                exc.__traceback__ if exc is not None else None,
            )


def _connection_manager(client: AsyncOpenAI) -> Any:
    connect = getattr(client.responses, "connect", None)
    if connect is None:
        raise LLMifyError(
            "Responses WebSocket transport requires a newer OpenAI SDK. "
            "Install py-llmify[websocket]."
        )
    return connect(extra_headers=_websocket_headers(client))


def _raise_websocket_dependency_error(exc: OpenAIError) -> None:
    if "openai[realtime]" in str(exc):
        raise LLMifyError(
            "Responses WebSocket transport requires the 'websockets' package. "
            "Install py-llmify[websocket]."
        ) from exc


def _websocket_headers(client: AsyncOpenAI) -> dict[str, str]:
    """Forward resolved authentication and client headers to the handshake."""
    headers = {**client.auth_headers, **client.default_headers}
    return {
        key: value
        for key, value in headers.items()
        if isinstance(value, str) and key.lower() != "user-agent"
    }


class _WebSocketResponsesSession:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._response_ids: set[str] = set()

    async def events(self, request: dict[str, Any]) -> AsyncIterator[Any]:
        websocket_request = {"type": "response.create", **request}
        websocket_request.pop("stream", None)
        websocket_request.pop("background", None)
        extra_body = websocket_request.pop("extra_body", None)
        if isinstance(extra_body, dict):
            websocket_request.update(extra_body)
        await self._connection.send(websocket_request)

        while True:
            event = await self._connection.recv()
            yield event
            if isinstance(
                event,
                (
                    ResponseCompletedEvent,
                    ResponseIncompleteEvent,
                    ResponseFailedEvent,
                    ResponseErrorEvent,
                ),
            ):
                return

    def can_continue_from(self, response_id: str) -> bool:
        return response_id in self._response_ids

    def remember(self, response_id: str) -> None:
        self._response_ids.add(response_id)
