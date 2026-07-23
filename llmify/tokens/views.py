from pydantic import BaseModel
from tokenary import ModelName


class TokenUsageEntry(BaseModel):
    model: ModelName
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cached_tokens: int | None = None


class UsageSummary(BaseModel):
    entry_count: int
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_prompt_cached_tokens: int


class CostSummary(BaseModel):
    entry_count: int
    currency: str = "USD"
    input_cost: float = 0.0
    output_cost: float = 0.0
    reasoning_cost: float = 0.0
    audio_input_cost: float = 0.0
    image_cost: float = 0.0
    code_interpreter_cost: float = 0.0
    file_search_call_cost: float = 0.0
    file_search_storage_cost: float = 0.0
    vector_store_cost: float = 0.0
    total_cost: float = 0.0
