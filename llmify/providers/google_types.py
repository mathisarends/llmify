from llmify.views import (
    ChatInvokeCompletion,
    ChatInvokeUsage,
    StreamEnd,
    StreamTextDelta,
    StreamToolCall,
)


class GoogleUsage(ChatInvokeUsage):
    prompt_image_tokens: int | None = None
    """Prompt tokens attributed to image modality input."""


class GoogleCompletion[T](ChatInvokeCompletion[T]):
    usage: GoogleUsage | None = None


class GoogleStreamEnd(StreamEnd):
    usage: GoogleUsage | None = None


type GoogleStreamEvent = StreamTextDelta | StreamToolCall | GoogleStreamEnd
