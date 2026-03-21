import asyncio
from llmify import ChatOpenAI, SystemMessage, UserMessage
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
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
