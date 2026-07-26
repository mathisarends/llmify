"""Call a Codex endpoint with a borrowed ChatGPT session.

Put the values into your environment (see `.env.example`):

    CODEX_ACCESS_KEY   access token of the ChatGPT session
    CODEX_ACCOUNT_ID   account id sent as `ChatGPT-Account-Id` header
    CODEX_BASE_URL     endpoint the token is valid for
"""

import asyncio
import os

from dotenv import load_dotenv

from llmify import ChatOpenAI, SystemMessage, UserMessage

load_dotenv(override=True)


async def main() -> None:
    llm = ChatOpenAI(
        model=os.environ["CODEX_MODEL"],
        api_key=os.environ["CODEX_ACCESS_KEY"],
        base_url="https://chatgpt.com/backend-api/codex",
        default_headers={"ChatGPT-Account-Id": os.environ["CODEX_ACCOUNT_ID"]},
    )

    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2+2?"),
        ]
    )
    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
