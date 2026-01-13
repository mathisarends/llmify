import asyncio
from llmify import ChatOpenAI, UserMessage, ImageMessage


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    response = await llm.invoke(
        [
            UserMessage("What color is this image?"),
            ImageMessage(
                base64_data=red_pixel_base64, media_type="image/png", detail="high"
            ),
        ]
    )

    print(f"Image analysis: {response}")


if __name__ == "__main__":
    asyncio.run(main())
