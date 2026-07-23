import llmify
import pytest
from pydantic import ValidationError

from llmify.tokens import ModelName, TokenTracker, TokenUsageEntry, UsageSummary
from llmify.views import ChatInvokeCompletion, ChatInvokeUsage, StreamEnd


def make_usage(
    prompt: int = 10,
    completion: int = 5,
    cached: int | None = None,
) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        prompt_cached_tokens=cached,
    )


def test_tokens_are_not_exported_from_the_base_package() -> None:
    assert not hasattr(llmify, "TokenTracker")


class TestAddAcceptedInputs:
    def test_accepts_raw_usage(self) -> None:
        tracker = TokenTracker()
        tracker.add(make_usage(prompt=10, completion=5), model=ModelName.GPT_4O)

        assert tracker.entry_count == 1
        assert tracker.entries[0].total_tokens == 15

    def test_accepts_chat_invoke_completion(self) -> None:
        tracker = TokenTracker()
        completion = ChatInvokeCompletion(
            completion="hi", usage=make_usage(prompt=7, completion=3)
        )
        tracker.add(completion, model=ModelName.GPT_4O)

        assert tracker.entries[0].prompt_tokens == 7
        assert tracker.entries[0].completion_tokens == 3

    def test_accepts_stream_end(self) -> None:
        tracker = TokenTracker()
        end = StreamEnd(usage=make_usage(prompt=4, completion=6))
        tracker.add(end, model=ModelName.CLAUDE_SONNET_4_20250514)

        assert tracker.entries[0].total_tokens == 10
        assert tracker.entries[0].model is ModelName.CLAUDE_SONNET_4_20250514


class TestAddNoOps:
    def test_none_usage_is_ignored(self) -> None:
        tracker = TokenTracker()
        tracker.add(None, model=ModelName.GPT_4O)

        assert tracker.entry_count == 0

    def test_completion_without_usage_is_ignored(self) -> None:
        tracker = TokenTracker()
        tracker.add(
            ChatInvokeCompletion(completion="hi", usage=None),
            model=ModelName.GPT_4O,
        )

        assert tracker.entry_count == 0

    def test_stream_end_without_usage_is_ignored(self) -> None:
        tracker = TokenTracker()
        tracker.add(StreamEnd(usage=None), model=ModelName.GPT_4O)

        assert tracker.entry_count == 0


class TestTagging:
    def test_rejects_unknown_model(self) -> None:
        with pytest.raises(ValidationError):
            TokenUsageEntry(
                model="not-in-tokenary",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )

    def test_entry_is_tagged_with_model(self) -> None:
        tracker = TokenTracker()
        tracker.add(make_usage(), model=ModelName.GPT_4O)
        tracker.add(make_usage(), model=ModelName.CLAUDE_SONNET_4_20250514)

        assert [e.model for e in tracker.entries] == [
            ModelName.GPT_4O,
            ModelName.CLAUDE_SONNET_4_20250514,
        ]

    def test_cached_tokens_preserved_on_entry(self) -> None:
        tracker = TokenTracker()
        tracker.add(make_usage(cached=4), model=ModelName.GPT_4O)

        assert tracker.entries[0].prompt_cached_tokens == 4


class TestSummary:
    def test_empty_tracker_summary_is_zeroed(self) -> None:
        summary = TokenTracker().summary()

        assert summary == UsageSummary(
            entry_count=0,
            total_tokens=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_prompt_cached_tokens=0,
        )

    def test_aggregates_across_entries_and_models(self) -> None:
        tracker = TokenTracker()
        tracker.add(
            make_usage(prompt=10, completion=5, cached=2),
            model=ModelName.GPT_4O,
        )
        tracker.add(
            make_usage(prompt=20, completion=8, cached=3),
            model=ModelName.CLAUDE_SONNET_4_20250514,
        )

        summary = tracker.summary()

        assert summary.entry_count == 2
        assert summary.total_prompt_tokens == 30
        assert summary.total_completion_tokens == 13
        assert summary.total_tokens == 43
        assert summary.total_prompt_cached_tokens == 5

    def test_treats_missing_cached_tokens_as_zero(self) -> None:
        tracker = TokenTracker()
        tracker.add(make_usage(cached=None), model=ModelName.GPT_4O)
        tracker.add(make_usage(cached=4), model=ModelName.GPT_4O)

        assert tracker.summary().total_prompt_cached_tokens == 4


class TestStateManagement:
    def test_total_tokens_property_matches_summary(self) -> None:
        tracker = TokenTracker()
        tracker.add(
            make_usage(prompt=10, completion=5),
            model=ModelName.GPT_4O,
        )
        tracker.add(make_usage(prompt=1, completion=1), model=ModelName.GPT_4O)

        assert tracker.total_tokens == tracker.summary().total_tokens == 17

    def test_reset_clears_all_entries(self) -> None:
        tracker = TokenTracker()
        tracker.add(make_usage(), model=ModelName.GPT_4O)
        tracker.reset()

        assert tracker.entry_count == 0
        assert tracker.summary().total_tokens == 0

    def test_entries_returns_a_copy(self) -> None:
        tracker = TokenTracker()
        tracker.add(make_usage(), model=ModelName.GPT_4O)

        entries = tracker.entries
        entries.append(
            TokenUsageEntry(
                model=ModelName.GPT_4O,
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
            )
        )

        assert tracker.entry_count == 1
