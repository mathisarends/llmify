import asyncio

from dotenv import load_dotenv

from llmify import ChatOpenAI, SystemMessage, UserMessage

load_dotenv(override=True)


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is 2+2?"),
        ]
    )

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
