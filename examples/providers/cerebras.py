import asyncio

from llmify import ChatCerebras, SystemMessage, UserMessage
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
    llm = ChatCerebras(model="gpt-oss-120b")
    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Explain why low-latency inference matters."),
        ]
    )
    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
