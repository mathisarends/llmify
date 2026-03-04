from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


def _truncate(text: str, max_length: int = 50) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class _MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


SupportedImageMediaType = Literal["image/jpeg", "image/png", "image/gif", "image/webp"]
_DetailLevel = Literal["auto", "low", "high"]


class ContentPartTextParam(BaseModel):
    text: str
    type: Literal["text"] = "text"

    def __str__(self) -> str:
        return f"Text: {_truncate(self.text)}"

    def __repr__(self) -> str:
        return f"ContentPartTextParam(text={_truncate(self.text)})"


class ImageURL(BaseModel):
    url: str
    detail: _DetailLevel = "auto"
    media_type: SupportedImageMediaType = "image/png"

    def __str__(self) -> str:
        url_display = (
            "<base64>" if self.url.startswith("data:") else _truncate(self.url)
        )
        return f"🖼️  Image[{self.media_type}, detail={self.detail}]: {url_display}"

    def __repr__(self) -> str:
        url_display = (
            "<base64>" if self.url.startswith("data:") else _truncate(self.url, 30)
        )
        return (
            f"ImageURL(url={repr(url_display)}, "
            f"detail={repr(self.detail)}, media_type={repr(self.media_type)})"
        )


class ContentPartImageParam(BaseModel):
    image_url: ImageURL
    type: Literal["image_url"] = "image_url"

    def __str__(self) -> str:
        return str(self.image_url)

    def __repr__(self) -> str:
        return f"ContentPartImageParam(image_url={repr(self.image_url)})"


class Function(BaseModel):
    arguments: str
    name: str

    def __str__(self) -> str:
        return f"{self.name}({_truncate(self.arguments, 80)})"

    def __repr__(self) -> str:
        return f"Function(name={repr(self.name)}, arguments={_truncate(repr(self.arguments), 50)})"


class ToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"] = "function"
    # Parsed tool model — populated by the caller after deserialisation
    tool: BaseModel | None = Field(default=None, exclude=True)

    def __str__(self) -> str:
        return f"ToolCall[{self.id}]: {self.function}"

    def __repr__(self) -> str:
        return f"ToolCall(id={repr(self.id)}, function={repr(self.function)})"


class _MessageBase(BaseModel):
    role: _MessageRole


class SystemMessage(_MessageBase):
    role: _MessageRole = _MessageRole.SYSTEM
    content: str | list[ContentPartTextParam]
    name: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content if part.type == "text")

    def __str__(self) -> str:
        return f"SystemMessage(content={_truncate(self.text)})"

    def __repr__(self) -> str:
        return f"SystemMessage(content={repr(_truncate(self.text))})"


class UserMessage(_MessageBase):
    role: _MessageRole = _MessageRole.USER
    content: str | list[ContentPartTextParam | ContentPartImageParam]
    name: str | None = None

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "\n".join(part.text for part in self.content if part.type == "text")

    def __str__(self) -> str:
        return f"UserMessage(content={_truncate(self.text)})"

    def __repr__(self) -> str:
        return f"UserMessage(content={repr(_truncate(self.text))})"


class AssistantMessage(_MessageBase):
    role: _MessageRole = _MessageRole.ASSISTANT
    content: str | list[ContentPartTextParam] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = []

    @property
    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list):
            return "".join(part.text for part in self.content if part.type == "text")
        return ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __str__(self) -> str:
        return f"AssistantMessage(content={_truncate(self.text)})"

    def __repr__(self) -> str:
        return f"AssistantMessage(content={repr(_truncate(self.text))})"


class ToolResultMessage(_MessageBase):
    """Carries the result of a tool call back to the model."""

    role: _MessageRole = _MessageRole.TOOL
    tool_call_id: str
    content: str

    def __str__(self) -> str:
        return f"ToolResultMessage(id={self.tool_call_id}, content={_truncate(self.content)})"

    def __repr__(self) -> str:
        return f"ToolResultMessage(tool_call_id={repr(self.tool_call_id)}, content={repr(_truncate(self.content))})"


class ModelResponse(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] = []
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_message(self) -> AssistantMessage:
        return AssistantMessage(content=self.content, tool_calls=self.tool_calls)

    def __str__(self) -> str:
        return (
            f"ModelResponse(finish_reason={self.finish_reason}, "
            f"tool_calls={len(self.tool_calls)}, content={_truncate(self.content or '')})"
        )

    def __repr__(self) -> str:
        return (
            f"ModelResponse(finish_reason={repr(self.finish_reason)}, "
            f"tool_calls={repr(self.tool_calls)}, content={repr(_truncate(self.content or ''))})"
        )


type Message = UserMessage | SystemMessage | AssistantMessage | ToolResultMessage
