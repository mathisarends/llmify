import pytest

from llmify.tokens import CostSummary, ModelName, TokenTracker
from llmify.views import ChatInvokeCompletion, ChatInvokeUsage

tokenary = pytest.importorskip("tokenary")
cost_api = pytest.importorskip("llmify.tokens.costs")
calculate_cost = cost_api.calculate_cost
calculate_costs = cost_api.calculate_costs


def make_usage(prompt: int = 1_000, completion: int = 500) -> ChatInvokeUsage:
    return ChatInvokeUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def test_calculate_cost_matches_tokenary() -> None:
    usage = make_usage()

    result = calculate_cost(usage, model=ModelName.GPT_4O)
    expected = tokenary.calculate(
        model=ModelName.GPT_4O,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
    )

    assert result == expected


def test_calculate_cost_rejects_missing_usage() -> None:
    completion = ChatInvokeCompletion(completion="hello", usage=None)

    with pytest.raises(ValueError, match="without usage"):
        calculate_cost(completion, model=ModelName.GPT_4O)


def test_empty_tracker_cost_summary_is_zeroed() -> None:
    assert TokenTracker().cost_summary() == CostSummary(entry_count=0)


def test_calculate_costs_aggregates_a_single_model_history() -> None:
    history = [
        ChatInvokeCompletion(completion="one", usage=make_usage()),
        ChatInvokeCompletion(
            completion="two",
            usage=make_usage(prompt=200, completion=100),
        ),
        ChatInvokeCompletion(completion="missing", usage=None),
    ]

    summary = calculate_costs(history, model=ModelName.GPT_4O)
    individual_costs = [
        calculate_cost(item, model=ModelName.GPT_4O)
        for item in history
        if item.usage is not None
    ]

    assert summary.entry_count == 2
    assert summary.total_cost == pytest.approx(
        sum(cost.total_cost for cost in individual_costs)
    )


def test_calculate_costs_uses_models_from_tagged_history() -> None:
    tracker = TokenTracker()
    tracker.add(make_usage(), model=ModelName.GPT_4O)
    tracker.add(
        make_usage(prompt=200, completion=100),
        model=ModelName.GPT_4O,
    )

    assert calculate_costs(tracker.entries) == tracker.cost_summary()


def test_tracker_calculates_per_entry_and_aggregate_costs() -> None:
    tracker = TokenTracker()
    tracker.add(make_usage(), model=ModelName.GPT_4O)
    tracker.add(
        make_usage(prompt=200, completion=100),
        model=ModelName.GPT_4O,
    )

    costs = tracker.costs()
    summary = tracker.cost_summary()

    assert len(costs) == 2
    assert summary.entry_count == 2
    assert summary.currency == "USD"
    assert summary.input_cost == pytest.approx(sum(cost.input_cost for cost in costs))
    assert summary.output_cost == pytest.approx(sum(cost.output_cost for cost in costs))
    assert summary.total_cost == pytest.approx(sum(cost.total_cost for cost in costs))
