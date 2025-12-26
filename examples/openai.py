import asyncio
from llmify.messages import SystemMessage, UserMessage
from llmify.providers.openai import OpenAILLM
from pydantic import BaseModel

async def simple_example():
    llm = OpenAILLM(model="gpt-4o")
    
    response = await llm.invoke([
        SystemMessage("Du bist ein hilfreicher Assistent"),
        UserMessage("Was ist 2+2?")
    ])
    
    print(response)


class Person(BaseModel):
    name: str
    age: int
    occupation: str

async def structured_example():
    llm = OpenAILLM(model="gpt-4o")
    
    person = await llm.invoke_structured([
        UserMessage("Extract info: Anna ist 28 Jahre alt und arbeitet als Softwareentwicklerin")
    ], response_model=Person)
    
    print(f"Name: {person.name}, Age: {person.age}, Job: {person.occupation}")


# Streaming
async def streaming_example():
    llm = OpenAILLM(model="gpt-4o")
    
    print("Streaming response: ", end="", flush=True)
    async for chunk in llm.stream([
        UserMessage("Erzähl mir einen kurzen Witz")
    ]):
        print(chunk, end="", flush=True)
    print()


# Alles zusammen
async def main():
    print("=== Simple Example ===")
    await simple_example()
    
    print("\n=== Structured Output ===")
    await structured_example()
    
    print("\n=== Streaming ===")
    await streaming_example()


if __name__ == "__main__":
    asyncio.run(main())