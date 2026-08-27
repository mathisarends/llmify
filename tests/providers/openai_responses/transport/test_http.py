from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llmify.providers.openai_responses.transport.http import (
    _http_request,
    _HTTPResponsesSession,
)


async def _stream(*events):
    for event in events:
        yield event


def test_http_request_enables_streaming_without_mutating_input() -> None:
    request = {
        "model": "gpt-test",
        "extra_body": {"existing": True},
        "prompt_cache_options": {"retention": "24h"},
    }

    result = _http_request(request)

    assert request["prompt_cache_options"] == {"retention": "24h"}
    assert result == {
        "model": "gpt-test",
        "stream": True,
        "extra_body": {
            "existing": True,
            "prompt_cache_options": {"retention": "24h"},
        },
    }


@pytest.mark.asyncio
async def test_http_session_streams_response_events() -> None:
    create = AsyncMock(return_value=_stream("first", "second"))
    session = _HTTPResponsesSession(
        SimpleNamespace(responses=SimpleNamespace(create=create))
    )

    events = [event async for event in session.events({"model": "gpt-test"})]

    assert events == ["first", "second"]
    create.assert_awaited_once_with(model="gpt-test", stream=True)
    assert session.can_continue_from("response-1") is False
    assert session.remember("response-1") is None
