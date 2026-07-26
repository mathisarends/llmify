import asyncio

from dotenv import load_dotenv

from llmify import ChatOpenAIResponses, SystemMessage, UserMessage

load_dotenv(override=True)


async def main() -> None:
    llm = ChatOpenAIResponses(model="gpt-5.4-mini")

    response = await llm.invoke(
        [
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="What is the OpenAI Responses API?"),
        ]
    )

    print(response.completion)


if __name__ == "__main__":
    asyncio.run(main())
