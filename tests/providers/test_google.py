import json
from types import SimpleNamespace

import pytest

pytest.importorskip("google.genai")

from llmify.base import ChatModel
from llmify.providers.google import ChatGoogle


class MockGoogleModel(ChatGoogle):
    def __init__(self):
        ChatModel.__init__(self, model="gemini-test")
        self._client = SimpleNamespace(models=SimpleNamespace())


class TestGoogleResponseParsing:
    def test_parses_direct_function_calls(self) -> None:
        model = MockGoogleModel()
        response = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    id="call_123",
                    name="get_weather",
                    args={"city": "Berlin", "unit": "celsius"},
                )
            ]
        )

        tool_calls = model._parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"
        assert tool_calls[0].function.name == "get_weather"
        assert json.loads(tool_calls[0].function.arguments) == {
            "city": "Berlin",
            "unit": "celsius",
        }

    def test_parses_nested_function_call_parts_with_stable_fallback_id(self) -> None:
        model = MockGoogleModel()
        response = SimpleNamespace(
            function_calls=[
                SimpleNamespace(
                    function_call=SimpleNamespace(
                        name="search_web",
                        args={"query": "gemini function calling"},
                    )
                )
            ]
        )

        tool_calls = model._parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_0_search_web"
        assert tool_calls[0].function.name == "search_web"
        assert json.loads(tool_calls[0].function.arguments) == {
            "query": "gemini function calling"
        }

    def test_parses_usage_with_image_tokens(self) -> None:
        model = MockGoogleModel()
        usage = SimpleNamespace(
            prompt_token_count=11,
            cached_content_token_count=3,
            candidates_token_count=7,
            total_token_count=18,
            prompt_tokens_details=[
                SimpleNamespace(modality="TEXT", token_count=5),
                SimpleNamespace(modality="IMAGE", token_count=6),
            ],
        )

        parsed = model._parse_usage(usage)

        assert parsed is not None
        assert parsed.prompt_tokens == 11
        assert parsed.prompt_cached_tokens == 3
        assert parsed.prompt_image_tokens == 6
        assert parsed.completion_tokens == 7
        assert parsed.total_tokens == 18

    def test_parses_usage_total_when_missing(self) -> None:
        model = MockGoogleModel()
        usage = SimpleNamespace(
            prompt_token_count=4,
            candidates_token_count=9,
            total_token_count=None,
            prompt_tokens_details=[],
        )

        parsed = model._parse_usage(usage)

        assert parsed is not None
        assert parsed.total_tokens == 13

    def test_parses_stop_reason(self) -> None:
        model = MockGoogleModel()
        response = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])

        assert model._stop_reason(response) == "STOP"
