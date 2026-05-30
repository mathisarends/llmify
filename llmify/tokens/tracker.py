from __future__ import annotations

from llmify.tokens.views import TokenUsageEntry, UsageSummary
from llmify.views import ChatInvokeCompletion, ChatInvokeUsage, StreamEnd

# Anything that carries (or is) a usage record can be handed to the tracker.
type Trackable = ChatInvokeUsage | ChatInvokeCompletion | StreamEnd | None


def _resolve_usage(usage: Trackable) -> ChatInvokeUsage | None:
    if usage is None:
        return None
    if isinstance(usage, ChatInvokeUsage):
        return usage
    # ChatInvokeCompletion / StreamEnd both expose `.usage`.
    return usage.usage


class TokenTracker:
    """Collects token usage across many calls and aggregates it on demand.

    The tracker is provider-agnostic: hand it the ``usage`` from any response
    (or the response/stream-end object itself) together with the model name.
    """

    def __init__(self) -> None:
        self._entries: list[TokenUsageEntry] = []

    def add(self, usage: Trackable, model: str) -> None:
        """Record a usage event for ``model``. No-ops if there is nothing to record."""
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
        return list(self._entries)

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
