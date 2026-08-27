"""Call Codex through the Responses API WebSocket transport.

Install the optional transport dependency with ``uv sync --extra websocket``
and authenticate the local Codex CLI with ``codex login`` first.
"""

import asyncio

from llmify import (
    ChatCodex,
    ResponsesOptions,
    SystemMessage,
    UserMessage,
    WebSocketResponsesTransport,
)


async def main() -> None:
    llm = ChatCodex.from_cli(
        model="gpt-5.6-terra",
        reasoning_effort="high",
        transport=WebSocketResponsesTransport(),
        responses_options=ResponsesOptions(reasoning_summary="concise"),
    )

    response = await llm.invoke(
        [
            SystemMessage(content="You are a concise coding assistant."),
            UserMessage(content="Explain in two sentences why WebSockets are useful."),
        ]
    )

    print(response.completion)
    print(response.usage)


if __name__ == "__main__":
    asyncio.run(main())
