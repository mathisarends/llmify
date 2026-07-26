"""Call Codex with a borrowed ChatGPT session.

Put the values into your environment (see `.env.example`):

    CODEX_ACCESS_KEY   access token of the ChatGPT session
    CODEX_ACCOUNT_ID   account id sent as `ChatGPT-Account-Id` header
    CODEX_MODEL        model slug to call
"""

import asyncio
import os

from dotenv import load_dotenv

from llmify import ChatCodex, SystemMessage, UserMessage

load_dotenv(override=True)


async def main() -> None:
    llm = ChatCodex(
        model=os.environ["CODEX_MODEL"],
        api_key=os.environ["CODEX_ACCESS_KEY"],
        chatgpt_account_id=os.environ["CODEX_ACCOUNT_ID"],
    )

    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(
                content="Was würdest du sagen ist das Geheimnis für die plätzliche Verbesserung von Coding Agents"
            ),
        ]
    )
    print(response.completion)
    print(response.usage)


if __name__ == "__main__":
    asyncio.run(main())
