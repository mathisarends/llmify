"""HTTP streaming transport for the Responses API."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from openai import AsyncOpenAI

from .base import ResponsesSession


class HTTPResponsesTransport:
    @asynccontextmanager
    async def session(
        self, client: AsyncOpenAI
    ) -> AsyncGenerator[ResponsesSession, None]:
        yield _HTTPResponsesSession(client)


class _HTTPResponsesSession:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def events(self, request: dict[str, Any]) -> AsyncIterator[Any]:
        stream = await self._client.responses.create(**_http_request(request))
        async for event in stream:
            yield event

    def can_continue_from(self, response_id: str) -> bool:
        return False

    def remember(self, response_id: str) -> None:
        pass


def _http_request(request: dict[str, Any]) -> dict[str, Any]:
    request = {**request, "stream": True}
    prompt_cache_options = request.pop("prompt_cache_options", None)
    if prompt_cache_options is not None:
        extra_body = dict(request.get("extra_body") or {})
        extra_body["prompt_cache_options"] = prompt_cache_options
        request["extra_body"] = extra_body
    return request
