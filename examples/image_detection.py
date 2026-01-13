import asyncio
from llmify import ChatOpenAI, UserMessage, ImageMessage


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
    print("=== Image Analysis (inline) ===")
    await image_example_inline()


if __name__ == "__main__":
    asyncio.run(main())
