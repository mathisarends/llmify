import asyncio
from llmify import ChatOpenAI, SystemMessage, UserMessage


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke(
        [SystemMessage("You are a helpful assistant."), UserMessage("What is 2+2?")]
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())
