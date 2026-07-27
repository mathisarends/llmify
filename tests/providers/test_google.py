import json

import pytest

pytest.importorskip("google.genai")

from google.genai import types

from llmify import AssistantMessage, ToolResultMessage
from llmify.messages import Function, ToolCall
from llmify.providers.google import (
    _convert_messages,
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
