from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from openai.types.responses import Response, ResponseCompletedEvent

from llmify.providers.openai_responses.transport.websocket import (
    WebSocketResponsesTransport,
    _websocket_headers,
    _WebSocketResponsesSession,
)


def _completed() -> ResponseCompletedEvent:
    response = Response.model_construct(
        id="response-1",
        status="completed",
        incomplete_details=None,
        error=None,
        usage=None,
        output=[],
    )
    return ResponseCompletedEvent.model_construct(
        response=response,
        sequence_number=0,
        type="response.completed",
    )


def test_websocket_headers_merge_auth_and_defaults() -> None:
    client = SimpleNamespace(
        auth_headers={"Authorization": "Bearer token", "User-Agent": "sdk"},
        default_headers={"ChatGPT-Account-Id": "account-1", "Ignored": 42},
    )

    assert _websocket_headers(client) == {
        "Authorization": "Bearer token",
        "ChatGPT-Account-Id": "account-1",
    }


@pytest.mark.asyncio
async def test_websocket_session_normalizes_request_and_tracks_responses() -> None:
    completed = _completed()
    connection = SimpleNamespace(
        send=AsyncMock(),
        recv=AsyncMock(return_value=completed),
    )
    session = _WebSocketResponsesSession(connection)
    request = {
        "model": "gpt-test",
        "stream": True,
        "background": False,
        "extra_body": {"prompt_cache_options": {"retention": "24h"}},
    }

    events = [event async for event in session.events(request)]

    assert events == [completed]
    connection.send.assert_awaited_once_with(
        {
            "type": "response.create",
            "model": "gpt-test",
            "prompt_cache_options": {"retention": "24h"},
        }
    )
    assert session.can_continue_from("response-1") is False
    session.remember("response-1")
    assert session.can_continue_from("response-1") is True


class _ConnectionManager:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self.connection

    async def __aexit__(self, *_args):
        self.exit_count += 1


@pytest.mark.asyncio
async def test_prewarm_opens_without_sending_and_reuses_until_closed() -> None:
    connection = SimpleNamespace(send=AsyncMock(), recv=AsyncMock())
    manager = _ConnectionManager(connection)
    client = SimpleNamespace(
        responses=SimpleNamespace(connect=Mock(return_value=manager)),
        auth_headers={"Authorization": "Bearer token"},
        default_headers={},
    )
    transport = WebSocketResponsesTransport()

    await transport.prewarm(client)
    await transport.prewarm(client)

    assert transport.is_prewarmed is True
    assert manager.enter_count == 1
    connection.send.assert_not_awaited()

    async with transport.session(client) as first:
        pass
    async with transport.session(client) as second:
        pass

    assert first is second
    assert manager.exit_count == 0

    await transport.aclose()

    assert transport.is_prewarmed is False
    assert manager.exit_count == 1


@pytest.mark.asyncio
async def test_failed_prewarmed_session_is_not_reused() -> None:
    connection = SimpleNamespace(send=AsyncMock(), recv=AsyncMock())
    manager = _ConnectionManager(connection)
    client = SimpleNamespace(
        responses=SimpleNamespace(connect=Mock(return_value=manager)),
        auth_headers={},
        default_headers={},
    )
    transport = WebSocketResponsesTransport()
    await transport.prewarm(client)

    with pytest.raises(RuntimeError, match="request failed"):
        async with transport.session(client):
            raise RuntimeError("request failed")

    assert transport.is_prewarmed is False
    assert manager.exit_count == 1
