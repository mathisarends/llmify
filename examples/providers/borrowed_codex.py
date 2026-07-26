"""Call a Codex endpoint with a borrowed ChatGPT session.

The Codex backend only speaks the Responses API, so this uses `OpenAIResponses`
instead of `ChatOpenAI`. Put the values into your environment (see `.env.example`):

    CODEX_ACCESS_KEY   access token of the ChatGPT session
    CODEX_ACCOUNT_ID   account id sent as `ChatGPT-Account-Id` header
    CODEX_MODEL        model slug to call
"""

import asyncio
import os

from dotenv import load_dotenv

from llmify import OpenAIResponses, SystemMessage, UserMessage

load_dotenv(override=True)

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


async def main() -> None:
    llm = OpenAIResponses(
        model=os.environ["CODEX_MODEL"],
        api_key=os.environ["CODEX_ACCESS_KEY"],
        base_url=CODEX_BASE_URL,
        default_headers={"ChatGPT-Account-Id": os.environ["CODEX_ACCOUNT_ID"]},
    )

    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2+2?"),
        ]
    )
    print(response.completion)
    print(response.usage)


if __name__ == "__main__":
    asyncio.run(main())
