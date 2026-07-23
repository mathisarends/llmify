import os
import httpx
from enum import StrEnum
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for ChatOpenAI. "
        "Install it with: pip install py-llmify[openai]"
    )

from llmify.providers.openai_compatible import OpenAICompatible


class OpenAIModel(StrEnum):
    GPT_5_6_SOL = "gpt-5.6-sol"
    GPT_5_6_TERRA = "gpt-5.6-terra"
    GPT_5_6_LUNA = "gpt-5.6-luna"

    GPT_5_5 = "gpt-5.5"
    GPT_5_5_PRO = "gpt-5.5-pro"

    GPT_5_4 = "gpt-5.4"
    GPT_5_4_PRO = "gpt-5.4-pro"
    GPT_5_4_MINI = "gpt-5.4-mini"
    GPT_5_4_NANO = "gpt-5.4-nano"

    GPT_5_3_CODEX = "gpt-5.3-codex"


class ChatOpenAI(OpenAICompatible):
    def __init__(
        self,
        model: str | OpenAIModel = OpenAIModel.GPT_5_6_TERRA,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        response_format: dict | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            seed=seed,
            response_format=response_format,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self._client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
