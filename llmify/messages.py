import base64
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class _MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


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


@dataclass
class ImageMessage(Message):
    base64_data: str
    media_type: _MediaType

    def __init__(
        self,
        base64_data: str,
        media_type: _MediaType | None = None,
        text: str | None = None,
    ):
        self.base64_data = base64_data

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
