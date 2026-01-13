import asyncio
from llmify import ChatOpenAI, UserMessage
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int
    occupation: str


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    structured_llm = llm.with_structured_output(Person)

    person: Person = await structured_llm.invoke(
        [UserMessage("Extract: Anna is 28 years old and works as a Software Engineer")]
    )

    print(f"Name: {person.name}, Age: {person.age}, Job: {person.occupation}")


if __name__ == "__main__":
    asyncio.run(main())
