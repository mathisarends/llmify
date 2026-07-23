from __future__ import annotations

from typing import TYPE_CHECKING

from tokenary import ModelName

from llmify.tokens.views import CostSummary, TokenUsageEntry, UsageSummary
from llmify.views import ChatInvokeCompletion, ChatInvokeUsage, StreamEnd

if TYPE_CHECKING:
    from tokenary import CostBreakdown

type Trackable = ChatInvokeUsage | ChatInvokeCompletion | StreamEnd | None


def _resolve_usage(usage: Trackable) -> ChatInvokeUsage | None:
    if usage is None:
        return None
    if isinstance(usage, ChatInvokeUsage):
        return usage
    return usage.usage


class TokenTracker:
    def __init__(self) -> None:
        self._entries: list[TokenUsageEntry] = []

    def add(self, usage: Trackable, model: ModelName) -> None:
        resolved = _resolve_usage(usage)
        if resolved is None:
            return

        self._entries.append(
            TokenUsageEntry(
                model=model,
                prompt_tokens=resolved.prompt_tokens,
                completion_tokens=resolved.completion_tokens,
                total_tokens=resolved.total_tokens,
                prompt_cached_tokens=resolved.prompt_cached_tokens,
            )
        )

    def reset(self) -> None:
        self._entries = []

    @property
    def entries(self) -> list[TokenUsageEntry]:
        return [entry.model_copy() for entry in self._entries]

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self._entries)

    def summary(self) -> UsageSummary:
        return UsageSummary(
            entry_count=len(self._entries),
            total_tokens=sum(e.total_tokens for e in self._entries),
            total_prompt_tokens=sum(e.prompt_tokens for e in self._entries),
            total_completion_tokens=sum(e.completion_tokens for e in self._entries),
            total_prompt_cached_tokens=sum(
                e.prompt_cached_tokens or 0 for e in self._entries
            ),
        )

    def costs(self) -> list[CostBreakdown]:
        from llmify.tokens.costs import calculate_cost

        return [calculate_cost(entry) for entry in self._entries]

    def cost_summary(self) -> CostSummary:
        from llmify.tokens.costs import calculate_costs

        return calculate_costs(self._entries)
