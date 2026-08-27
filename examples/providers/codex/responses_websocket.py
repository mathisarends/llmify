"""Prewarm a Codex Responses API WebSocket before the first request.

Install the optional transport dependency with ``uv sync --extra websocket``
and authenticate the local Codex CLI with ``codex login`` first.
"""

import asyncio

from llmify import (
    ChatCodex,
    SystemMessage,
    UserMessage,
    WebSocketResponsesTransport,
)


async def main() -> None:
    async with ChatCodex.from_cli(
        model="gpt-5.3-codex-spark",
        transport=WebSocketResponsesTransport(),
    ) as llm:
        await llm.prewarm()

        response = await llm.invoke(
            [
                SystemMessage(content="You are a concise coding assistant."),
                UserMessage(content="Why does WebSocket prewarming reduce latency?"),
            ]
        )

        print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
