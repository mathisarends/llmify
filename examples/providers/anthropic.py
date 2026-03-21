import asyncio
from llmify import ChatAnthropic, SystemMessage, UserMessage
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
    llm = ChatAnthropic(model="claude-sonnet-4-20250514")

    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2+2?"),
        ]
    )

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
