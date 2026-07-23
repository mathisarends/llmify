from __future__ import annotations

from collections.abc import Iterable

try:
    from tokenary import CostBreakdown, ModelName, calculate
except ModuleNotFoundError as exc:
    if exc.name != "tokenary":
        raise
    raise ModuleNotFoundError(
        "Cost calculation requires the 'tokens' extra. "
        "Install it with: pip install 'py-llmify[tokens]'"
    ) from exc

from llmify.tokens.tracker import Trackable, _resolve_usage
from llmify.tokens.views import CostSummary, TokenUsageEntry

type Costable = Trackable | TokenUsageEntry


def calculate_cost(
    usage: Costable,
    *,
    model: ModelName | None = None,
) -> CostBreakdown:
    if isinstance(usage, TokenUsageEntry):
        model = model or usage.model
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
    else:
        if model is None:
            raise TypeError("model is required for untagged usage")
        resolved = _resolve_usage(usage)
        if resolved is None:
            raise ValueError("Cannot calculate costs without usage information")
        prompt_tokens = resolved.prompt_tokens
        completion_tokens = resolved.completion_tokens

    return calculate(
        model=model,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
    )


def calculate_costs(
    history: Iterable[Costable],
    *,
    model: ModelName | None = None,
) -> CostSummary:
    costs: list[CostBreakdown] = []
    for usage in history:
        if not isinstance(usage, TokenUsageEntry) and _resolve_usage(usage) is None:
            continue
        costs.append(calculate_cost(usage, model=model))

    return _summarize(costs)


def _summarize(costs: list[CostBreakdown]) -> CostSummary:
    return CostSummary(
        entry_count=len(costs),
        input_cost=sum(cost.input_cost for cost in costs),
        output_cost=sum(cost.output_cost for cost in costs),
        reasoning_cost=sum(cost.reasoning_cost for cost in costs),
        audio_input_cost=sum(cost.audio_input_cost for cost in costs),
        image_cost=sum(cost.image_cost for cost in costs),
        code_interpreter_cost=sum(cost.code_interpreter_cost for cost in costs),
        file_search_call_cost=sum(cost.file_search_call_cost for cost in costs),
        file_search_storage_cost=sum(cost.file_search_storage_cost for cost in costs),
        vector_store_cost=sum(cost.vector_store_cost for cost in costs),
        total_cost=sum(cost.total_cost for cost in costs),
    )
