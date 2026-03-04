import os
from llmify import ChatAzureOpenAI, UserMessage
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("LITE_LLM_API_KEY")
azure_endpoint = os.getenv("AZURE_ENDPOINT")

async def main():
    llm = ChatAzureOpenAI(model="gpt-4o", api_key=api_key, azure_endpoint=azure_endpoint)

    response = await llm.invoke([UserMessage(content="What is the capital of France?")])

    print(response)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())