import os
from typing import Type, TypeVar, AsyncIterator, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, Content, Part
from llmify.messages import Message, ImageMessage, SystemMessage
from llmify.providers.base import BaseChatModel

T = TypeVar('T', bound=BaseModel)

load_dotenv(override=True)

class ChatGemini(BaseChatModel):
    def __init__(
        self, 
        model: str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        **kwargs: Any
    ):
        super().__init__(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            **kwargs
        )
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
        
        self._client = genai.Client(api_key=api_key)
        self._model = model
    
    async def invoke(
        self, 
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> str:
        contents = self._convert_messages(messages)
        config = self._build_generation_config(max_tokens, temperature, kwargs)
        
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config
        )
        return response.text

    def _convert_messages(self, messages: list[Message]) -> list[Content]:
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System message als user message mit prefix
                converted.append(Content(
                    role="user",
                    parts=[Part(text=f"[System Instructions] {msg.content}")]
                ))
                continue
            
            if isinstance(msg, ImageMessage):
                parts = []
                if msg.content:
                    parts.append(Part(text=msg.content))
                parts.append(Part(
                    inline_data={
                        "mime_type": msg.media_type.value,
                        "data": msg.base64_data
                    }
                ))
                role = "user" if msg.role == "user" else "model"
                converted.append(Content(role=role, parts=parts))
                continue
            
            role = "user" if msg.role == "user" else "model"
            converted.append(Content(
                role=role,
                parts=[Part(text=msg.content)]
            ))
        
        return converted
    
    async def invoke_structured(
        self, 
        messages: list[Message], 
        response_model: Type[T],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> T:
        contents = self._convert_messages(messages)
        config = self._build_generation_config(max_tokens, temperature, kwargs)
        config.response_mime_type = "application/json"
        config.response_schema = response_model
        
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config
        )
        return response_model.model_validate_json(response.text)
    
    async def stream(
        self, 
        messages: list[Message],
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any
    ) -> AsyncIterator[str]:
        contents = self._convert_messages(messages)
        config = self._build_generation_config(max_tokens, temperature, kwargs)
        
        async for chunk in self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config
        ):
            if chunk.text:
                yield chunk.text
    
    def _build_generation_config(
        self, 
        max_tokens: int | None, 
        temperature: float | None, 
        kwargs: dict[str, Any]
    ) -> GenerateContentConfig:
        params = self._merge_params({
            "max_tokens": max_tokens, 
            "temperature": temperature, 
            **kwargs
        })
        
        config_dict = {}
        if "max_tokens" in params:
            config_dict["max_output_tokens"] = params.pop("max_tokens")
        if "temperature" in params:
            config_dict["temperature"] = params.pop("temperature")
        if "top_p" in params:
            config_dict["top_p"] = params.pop("top_p")
        if "stop" in params:
            config_dict["stop_sequences"] = params.pop("stop")
        
        config_dict.update(params)
        return GenerateContentConfig(**config_dict)