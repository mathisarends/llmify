from pydantic import BaseModel


class TokenUsageEntry(BaseModel):
    """A single recorded usage event, tagged with the model that produced it."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cached_tokens: int | None = None


class UsageSummary(BaseModel):
    """Aggregated token usage across all recorded entries."""

    entry_count: int
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_prompt_cached_tokens: int
