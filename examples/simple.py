import asyncio
from llmify import ChatOpenAI, SystemMessage, UserMessage


async def simple_example():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke(
        [SystemMessage("You are a helpful assistant."), UserMessage("What is 2+2?")]
    )

    print(response)


async def main():
    print("=== Simple Completion ===")
    await simple_example()


if __name__ == "__main__":
    asyncio.run(main())
