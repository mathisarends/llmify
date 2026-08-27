from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai.types.responses import Response, ResponseCompletedEvent

from llmify.providers.openai_responses.transport.websocket import (
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
