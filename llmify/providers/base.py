from abc import ABC, abstractmethod
from typing import AsyncIterator, TypeVar, Type, Any
from pydantic import BaseModel
import httpx

from llmify.messages import Message, ImageMessage

T = TypeVar('T', bound=BaseModel)

class BaseChatModel(ABC):
    def __init__(
        self,
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
        **kwargs: Any
    ):
        self._default_max_tokens = max_tokens
        self._default_temperature = temperature
        self._default_top_p = top_p
        self._default_frequency_penalty = frequency_penalty
        self._default_presence_penalty = presence_penalty
        self._default_stop = stop
        self._default_seed = seed
        self._default_response_format = response_format
        self._default_timeout = timeout
        self._default_max_retries = max_retries
        self._default_kwargs = kwargs
    
    def _merge_params(self, method_kwargs: dict[str, Any]) -> dict[str, Any]:
        params = {**self._default_kwargs, **method_kwargs}
        
        param_mapping = {
            "max_tokens": self._default_max_tokens,
            "temperature": self._default_temperature,
            "top_p": self._default_top_p,
            "frequency_penalty": self._default_frequency_penalty,
            "presence_penalty": self._default_presence_penalty,
            "stop": self._default_stop,
            "seed": self._default_seed,
            "response_format": self._default_response_format,
        }
        
        for key, default_value in param_mapping.items():
            if key not in method_kwargs and default_value is not None:
                params[key] = default_value
        
        params = {k: v for k, v in params.items() if v is not None}
        return params
    
    @abstractmethod
    async def invoke(
        self, 
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> str:
        pass
    
    @abstractmethod
    async def invoke_structured(
        self, 
        messages: list[Message], 
        response_model: Type[T],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> T:
        pass
    
    @abstractmethod
    async def stream(
        self, 
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        pass


class BaseOpenAICompatible(BaseChatModel):
    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        converted = []
        for msg in messages:
            if not isinstance(msg, ImageMessage):
                converted.append({"role": msg.role, "content": msg.content})
                continue
            
            content = []
            if msg.content:
                content.append({"type": "text", "text": msg.content})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{msg.media_type};base64,{msg.base64_data}"
                }
            })
            converted.append({"role": msg.role, "content": content})
        
        return converted