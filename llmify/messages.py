import base64
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class _MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


_MediaType = Literal["image/jpeg", "image/png"]


@dataclass
class Message:
    role: _MessageRole
    content: str


class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__(role=_MessageRole.SYSTEM, content=content)


class UserMessage(Message):
    def __init__(self, content: str):
        super().__init__(role=_MessageRole.USER, content=content)


class AssistantMessage(Message):
    def __init__(self, content: str):
        super().__init__(role=_MessageRole.ASSISTANT, content=content)


_DetailLevel = Literal["low", "high", "auto"]


@dataclass
class ImageMessage(Message):
    base64_data: str
    media_type: _MediaType
    detail: _DetailLevel = "auto"

    def __init__(
        self,
        base64_data: str,
        media_type: _MediaType | None = None,
        text: str | None = None,
        detail: _DetailLevel = "auto",
    ):
        self.base64_data = base64_data
        self.detail = detail

        if media_type is None:
            self.media_type = self._detect_media_type(base64_data)
        else:
            self.media_type = media_type

        super().__init__(role=_MessageRole.USER, content=text or "")

    @staticmethod
    def _detect_media_type(base64_data: str) -> _MediaType:
        try:
            header = base64.b64decode(base64_data[:20])
            if header.startswith(b"\x89PNG"):
                return "image/png"
        except Exception:
            pass
        return "image/jpeg"


@dataclass
class ToolResultMessage(Message):
    tool_call_id: str

    def __init__(self, tool_call_id: str, content: str):
        self.tool_call_id = tool_call_id
        super().__init__(role=_MessageRole.TOOL, content=content)


@dataclass
class AssistantToolCallMessage(Message):
    tool_calls: list["ToolCall"]

    def __init__(self, content: str | None, tool_calls: list["ToolCall"]):
        self.tool_calls = tool_calls
        super().__init__(role=_MessageRole.ASSISTANT, content=content or "")


@dataclass
class ToolCall:
    id: str
    name: str
    tool: BaseModel


@dataclass
class ModelResponse:
    content: str | None
    tool_calls: list[ToolCall]
    finish_reason: Literal["stop", "tool_calls", "length", "content_filter"]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def to_message(self) -> Message:
        if self.has_tool_calls:
            return AssistantToolCallMessage(
                content=self.content, tool_calls=self.tool_calls
            )
        return AssistantMessage(content=self.content or "")
