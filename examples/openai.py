import asyncio
from llmify import ChatOpenAI, SystemMessage, UserMessage, ImageMessage
from pydantic import BaseModel


async def simple_example():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke(
        [SystemMessage("You are a helpful assistant."), UserMessage("What is 2+2?")]
    )

    print(response)


class Person(BaseModel):
    name: str
    age: int
    occupation: str


async def structured_example():
    llm = ChatOpenAI(model="gpt-4o")

    person = await llm.invoke_structured(
        [UserMessage("Extract: Anna is 28 years old and works as a Software Engineer")],
        response_model=Person,
    )

    print(f"Name: {person.name}, Age: {person.age}, Job: {person.occupation}")


async def streaming_example():
    llm = ChatOpenAI(model="gpt-4o")

    print("Streaming response: ", end="", flush=True)
    async for chunk in llm.stream([UserMessage("Tell me a short joke")]):
        print(chunk, end="", flush=True)
    print()


async def image_example_inline():
    llm = ChatOpenAI(model="gpt-4o")

    red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    response = await llm.invoke(
        [
            UserMessage("What color is this image?"),
            ImageMessage(base64_data=red_pixel_base64, media_type="image/png"),
        ]
    )

    print(f"Image analysis: {response}")


async def main():
    print("=== Simple Completion ===")
    await simple_example()

    print("\n=== Structured Output ===")
    await structured_example()

    print("\n=== Streaming ===")
    await streaming_example()

    print("\n=== Image Analysis (inline) ===")
    await image_example_inline()


if __name__ == "__main__":
    asyncio.run(main())
