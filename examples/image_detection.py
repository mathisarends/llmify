import asyncio
from llmify import (
    UserMessage,
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
)
from dotenv import load_dotenv

from llmify import ChatOpenAI

load_dotenv(override=True)


async def main():
    llm = ChatOpenAI(model="gpt-4o")

    red_pixel_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

    response = await llm.invoke(
        [
            UserMessage(
                content=[
                    ContentPartTextParam(text="What color is this image?"),
                    ContentPartImageParam(
                        image_url=ImageURL(
                            url=f"data:image/png;base64,{red_pixel_base64}",
                            media_type="image/png",
                            detail="high",
                        )
                    ),
                ]
            )
        ]
    )

    print(f"Image analysis: {response.completion}")


if __name__ == "__main__":
    asyncio.run(main())
