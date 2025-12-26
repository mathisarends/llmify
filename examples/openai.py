import asyncio
import base64
from pathlib import Path
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


async def image_example_file():
    llm = ChatOpenAI(model="gpt-4o")

    image_path = Path(__file__).parent / "test_image.jpg"
    base64_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    response = await llm.invoke(
        [
            UserMessage("Describe this image in detail"),
            ImageMessage(base64_data=base64_image, media_type="image/jpeg"),
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

    image_path = Path(__file__).parent / "test_image.jpg"
    if image_path.exists():
        print("\n=== Image Analysis (file) ===")
        await image_example_file()


if __name__ == "__main__":
    asyncio.run(main())
