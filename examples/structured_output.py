import asyncio
from llmify import ChatOpenAI, UserMessage
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)


class Person(BaseModel):
    name: str
    age: int
    occupation: str


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke(
        [
            UserMessage(
                content="Extract: Anna is 28 years old and works as a Software Engineer"
            )
        ],
        output_format=Person,
    )

    person = response.completion
    print(f"Name: {person.name}, Age: {person.age}, Job: {person.occupation}")


if __name__ == "__main__":
    asyncio.run(main())
