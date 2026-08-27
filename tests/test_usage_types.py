from llmify.providers.anthropic.types import (
    AnthropicCompletion,
    AnthropicStreamEnd,
    AnthropicUsage,
)
from llmify.providers.google.types import (
    GoogleCompletion,
    GoogleStreamEnd,
    GoogleUsage,
)
from llmify.providers.openai_responses.types import OpenAIResponsesUsage
from llmify.views import ChatInvokeUsage

COMMON_FIELDS = {
    "prompt_tokens",
    "prompt_cached_tokens",
    "completion_tokens",
    "total_tokens",
}


class TestChatInvokeUsage:
    def test_exposes_only_fields_every_provider_reports(self) -> None:
        assert set(ChatInvokeUsage.model_fields) == COMMON_FIELDS

    def test_provider_usage_extends_the_common_fields(self) -> None:
        assert set(AnthropicUsage.model_fields) - COMMON_FIELDS == {
            "prompt_cache_creation_tokens"
        }
        assert set(GoogleUsage.model_fields) - COMMON_FIELDS == {"prompt_image_tokens"}
        assert set(OpenAIResponsesUsage.model_fields) - COMMON_FIELDS == {
            "prompt_cache_write_tokens",
            "reasoning_tokens",
        }


class TestAnthropicViews:
    def test_completion_keeps_provider_usage(self) -> None:
        completion = AnthropicCompletion(
            completion="hi",
            usage=AnthropicUsage(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                prompt_cache_creation_tokens=4,
            ),
        )

        assert completion.usage is not None
        assert completion.usage.prompt_cache_creation_tokens == 4

    def test_stream_end_keeps_provider_usage(self) -> None:
        end = AnthropicStreamEnd(
            usage=AnthropicUsage(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                prompt_cache_creation_tokens=4,
            ),
        )

        assert end.usage is not None
        assert end.usage.prompt_cache_creation_tokens == 4


class TestGoogleViews:
    def test_completion_keeps_provider_usage(self) -> None:
        completion = GoogleCompletion(
            completion="hi",
            usage=GoogleUsage(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                prompt_image_tokens=6,
            ),
        )

        assert completion.usage is not None
        assert completion.usage.prompt_image_tokens == 6

    def test_stream_end_keeps_provider_usage(self) -> None:
        end = GoogleStreamEnd(
            usage=GoogleUsage(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
                prompt_image_tokens=6,
            ),
        )

        assert end.usage is not None
        assert end.usage.prompt_image_tokens == 6
