import pytest
from unittest.mock import AsyncMock, patch
from llmify import ChatOpenAI, UserMessage


@pytest.mark.asyncio
@patch("llmify.providers.openai.AsyncOpenAI")
async def test_invoke_with_mock(mock_client):
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="Hello!"))]
    mock_client.return_value.chat.completions.create = AsyncMock(
        return_value=mock_response
    )

    llm = ChatOpenAI(api_key="fake-key")
    response = await llm.invoke([UserMessage("Hi")])

    assert response == "Hello!"


@pytest.mark.asyncio
@patch("llmify.providers.openai.AsyncOpenAI")
async def test_streaming_with_mock(mock_client):
    chunk1 = AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="Hel"))])
    chunk2 = AsyncMock(choices=[AsyncMock(delta=AsyncMock(content="lo!"))])

    async def mock_stream(*args, **kwargs):
        yield chunk1
        yield chunk2

    mock_client.return_value.chat.completions.create = AsyncMock(
        return_value=mock_stream()
    )

    llm = ChatOpenAI(api_key="fake-key")
    chunks = []
    async for chunk in llm.stream([UserMessage("Hi")]):
        chunks.append(chunk)

    assert chunks == ["Hel", "lo!"]
