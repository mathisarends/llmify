import asyncio
import os
from llmify import ChatAzureOpenAI
from llmify.messages import (
    ContentPartImageParam,
    ContentPartTextParam,
    ImageURL,
    UserMessage,
)
from dotenv import load_dotenv

load_dotenv(override=True)


async def main():
    llm = ChatAzureOpenAI(
        model="gpt-4o",
        api_key=os.getenv("LITE_LLM_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    )
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
