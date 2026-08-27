"""Persistent WebSocket transport for the Responses API."""

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
    @asynccontextmanager
    async def session(
        self, client: AsyncOpenAI
    ) -> AsyncGenerator[ResponsesSession, None]:
        connect = getattr(client.responses, "connect", None)
        if connect is None:
            raise LLMifyError(
                "Responses WebSocket transport requires a newer OpenAI SDK. "
                "Install py-llmify[websocket]."
            )

        extra_headers = _websocket_headers(client)
        try:
            async with connect(extra_headers=extra_headers) as connection:
                yield _WebSocketResponsesSession(connection)
        except OpenAIError as exc:
            if "openai[realtime]" not in str(exc):
                raise
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
