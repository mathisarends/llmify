import asyncio
import os
from llmify import ChatAzureOpenAI, UserMessage
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

class Person(BaseModel):
    name: str
    age: int
    occupation: str

async def main():
    llm = ChatAzureOpenAI(
        model="gpt-4o",
        api_key=os.getenv("LITE_LLM_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    )
    response = await llm.invoke(
        [UserMessage(content="Extract: Anna is 28 years old and works as a Software Engineer")],
        response_model=Person,
    )

    print(f"Name: {response.completion.name}, Age: {response.completion.age}, Job: {response.completion.occupation}")

if __name__ == "__main__":
    asyncio.run(main())