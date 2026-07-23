from typing import TYPE_CHECKING

try:
    from tokenary import ModelName
except ModuleNotFoundError as exc:
    if exc.name != "tokenary":
        raise
    raise ModuleNotFoundError(
        "Token tracking requires the 'tokens' extra. "
        "Install it with: pip install 'py-llmify[tokens]'"
    ) from exc

from llmify.tokens.tracker import TokenTracker, Trackable
from llmify.tokens.views import CostSummary, TokenUsageEntry, UsageSummary

if TYPE_CHECKING:
    from llmify.tokens.costs import calculate_cost, calculate_costs

__all__ = [
    "calculate_cost",
    "calculate_costs",
    "CostSummary",
    "ModelName",
    "TokenTracker",
    "Trackable",
    "TokenUsageEntry",
    "UsageSummary",
]


def __getattr__(name: str):
    if name in {"calculate_cost", "calculate_costs"}:
        from llmify.tokens.costs import calculate_cost, calculate_costs

        return {
            "calculate_cost": calculate_cost,
            "calculate_costs": calculate_costs,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
