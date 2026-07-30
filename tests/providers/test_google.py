import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

pytest.importorskip("google.genai")

from google.genai import errors, types

from llmify import AssistantMessage, ToolResultMessage
from llmify import retries as retries_provider
from llmify.base import ChatModel
from llmify.messages import Function, ToolCall
from llmify.providers.google import (
    ChatGoogle,
    _convert_messages,
    _map_google_error,
    _parse_text,
    _parse_tool_calls,
    _parse_usage,
    _stop_reason,
)


class TestGoogleResponseParsing:
    def test_round_trips_function_call_metadata_and_result(self) -> None:
        messages = [
            AssistantMessage(
                tool_calls=[
                    ToolCall(
                        id="call_123",
                        function=Function(
                            name="get_weather",
                            arguments='{"city": "Berlin"}',
                        ),
                        provider_metadata={
                            "google": {"thought_signature": b"signature"}
                        },
                    )
                ]
            ),
            ToolResultMessage(
                tool_call_id="call_123",
                content="18 degrees Celsius",
            ),
        ]

        contents, _ = _convert_messages(messages)

        assert contents == [
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "id": "call_123",
                            "name": "get_weather",
                            "args": {"city": "Berlin"},
                        },
                        "thought_signature": b"signature",
                    }
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "id": "call_123",
                            "name": "get_weather",
                            "response": {"result": "18 degrees Celsius"},
                        }
                    }
                ],
            },
        ]

    def test_parses_text_alongside_function_calls(self) -> None:
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[
                            types.Part(text="Let me check. "),
                            types.Part(
                                function_call=types.FunctionCall(
                                    name="get_weather",
                                    args={"city": "Berlin"},
                                )
                            ),
                            types.Part(text="One moment."),
                        ]
                    )
                )
            ]
        )

        assert _parse_text(response) == "Let me check. One moment."

    def test_returns_empty_text_for_function_call_only(self) -> None:
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    name="get_weather",
                                    args={"city": "Berlin"},
                                )
                            )
                        ]
                    )
                )
            ]
        )

        assert _parse_text(response) == ""

    def test_excludes_thought_parts_from_text(self) -> None:
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[
                            types.Part(text="internal reasoning", thought=True),
                            types.Part(text="Visible answer"),
                        ]
                    )
                )
            ]
        )

        assert _parse_text(response) == "Visible answer"

    def test_parses_direct_function_calls(self) -> None:
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    id="call_123",
                                    name="get_weather",
                                    args={"city": "Berlin", "unit": "celsius"},
                                ),
                                thought_signature=b"signature",
                            )
                        ]
                    )
                )
            ]
        )

        tool_calls = _parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].provider_metadata == {
            "google": {"thought_signature": b"signature"}
        }
        assert json.loads(tool_calls[0].function.arguments) == {
            "city": "Berlin",
            "unit": "celsius",
        }

    def test_creates_stable_fallback_id(self) -> None:
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(
                        parts=[
                            types.Part(
                                function_call=types.FunctionCall(
                                    name="search_web",
                                    args={"query": "gemini function calling"},
                                )
                            )
                        ]
                    )
                )
            ]
        )

        tool_calls = _parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_0_search_web"
        assert tool_calls[0].function.name == "search_web"
        assert json.loads(tool_calls[0].function.arguments) == {
            "query": "gemini function calling"
        }

    def test_parses_usage_with_image_tokens(self) -> None:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=11,
            cached_content_token_count=3,
            candidates_token_count=7,
            total_token_count=18,
            prompt_tokens_details=[
                types.ModalityTokenCount(modality="TEXT", token_count=5),
                types.ModalityTokenCount(modality="IMAGE", token_count=6),
            ],
        )

        parsed = _parse_usage(usage)

        assert parsed is not None
        assert parsed.prompt_tokens == 11
        assert parsed.prompt_cached_tokens == 3
        assert parsed.prompt_image_tokens == 6
        assert parsed.completion_tokens == 7
        assert parsed.total_tokens == 18

    def test_parses_usage_total_when_missing(self) -> None:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=4,
            candidates_token_count=9,
            total_token_count=None,
            prompt_tokens_details=[],
        )

        parsed = _parse_usage(usage)

        assert parsed is not None
        assert parsed.total_tokens == 13

    def test_parses_stop_reason(self) -> None:
        response = types.GenerateContentResponse(
            candidates=[types.Candidate(finish_reason="STOP")]
        )

        assert _stop_reason(response) == "STOP"


class TestGoogleRetries:
    @staticmethod
    def _model(**kwargs) -> ChatGoogle:
        model = object.__new__(ChatGoogle)
        ChatModel.__init__(model, model="gemini-test", **kwargs)
        model._client = SimpleNamespace(models=SimpleNamespace())
        return model

    def test_disables_sdk_retries(self) -> None:
        with patch("llmify.providers.google.genai.Client") as client:
            ChatGoogle(api_key="test-key", max_retries=4)

        http_options = client.call_args.kwargs["http_options"]
        assert http_options.retry_options.attempts == 1

    def test_maps_retry_after_header(self) -> None:
        response = httpx.Response(
            status_code=429,
            headers={"retry-after": "2.5"},
            request=httpx.Request("POST", "https://generativelanguage.googleapis.com"),
        )
        mapped = _map_google_error(
            errors.APIError(
                429,
                {"message": "rate limited"},
                response=response,
            )
        )

        assert mapped.retry_after == 2.5

    @pytest.mark.asyncio
    async def test_retries_invoke_and_calls_the_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        callback = AsyncMock()
        monkeypatch.setattr(retries_provider.asyncio, "sleep", AsyncMock())
        model = self._model(max_retries=1, on_retry=callback)
        response = types.GenerateContentResponse(
            candidates=[
                types.Candidate(
                    content=types.Content(parts=[types.Part(text="done")]),
                    finish_reason="STOP",
                )
            ]
        )
        model._client.models.generate_content = AsyncMock(
            side_effect=[
                errors.APIError(503, {"message": "unavailable"}),
                response,
            ]
        )

        result = await model.invoke([])

        assert result.completion == "done"
        assert model._client.models.generate_content.await_count == 2
        callback.assert_awaited_once()
