"""Call Codex with the session of the locally installed Codex CLI.

No environment variables needed — `codex login` is the setup step. The access
token is refreshed automatically when it approaches expiry.
"""

import asyncio

from llmify import ChatCodex, SystemMessage, UserMessage


async def main() -> None:
    llm = ChatCodex.from_cli(model="gpt-5.6-terra", reasoning_effort="high")

    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Explain OAuth refresh tokens in two sentences."),
        ]
    )
    print(response.completion)
    print(response.usage)


if __name__ == "__main__":
    asyncio.run(main())
