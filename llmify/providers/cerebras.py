import os
from enum import StrEnum
from typing import Any

import httpx

try:
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "The 'openai' package is required for ChatCerebras. "
        "Install it with: pip install py-llmify[cerebras]"
    )

from llmify.providers.openai_compatible import OpenAICompatible


class CerebrasModel(StrEnum):
    GPT_OSS_120B = "gpt-oss-120b"
    GEMMA_4_31B_PREVIEW = "gemma-4-31b"
    ZAI_GLM_4_7_PREVIEW = "zai-glm-4.7"


class ChatCerebras(OpenAICompatible):
    def __init__(
        self,
        model: str | CerebrasModel = CerebrasModel.GPT_OSS_120B,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
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
            api_key = os.getenv("CEREBRAS_API_KEY")

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.cerebras.ai/v1",
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
        )
