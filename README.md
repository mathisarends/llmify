# llmify

A lightweight, type-safe Python library for LLM chat completions. Inspired by LangChain's message API but simpler and less opinionated.

**Features:**
- 🎯 Simple, intuitive API for OpenAI and Azure OpenAI
- 📝 Type-safe structured outputs with Pydantic
- 🛠️ Built-in tool calling support
- 🌊 Async streaming
- 🖼️ Image analysis support
- ⚡ Minimal dependencies, maximum flexibility

## Installation
```bash
pip install py-llmify
```

## Quick Start
```python
import asyncio
from llmify import ChatOpenAI, UserMessage, SystemMessage

async def main():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke([
        SystemMessage("You are a helpful assistant"),
        UserMessage("What is 2+2?")
    ])

    print(response)  # "2+2 equals 4"

asyncio.run(main())
```

## Core Features

### Message Types

llmify provides LangChain-style message types for clean conversation management:
```python
from llmify import SystemMessage, UserMessage, AssistantMessage, ImageMessage

messages = [
    SystemMessage("You are a Python expert"),
    UserMessage("How do I read a file?"),
    AssistantMessage("You can use open() with a context manager"),
    UserMessage("Show me an example")
]
```

### Structured Outputs

Get type-safe, validated responses using Pydantic models:
```python
from pydantic import BaseModel
from llmify import ChatOpenAI, UserMessage

class Person(BaseModel):
    name: str
    age: int
    occupation: str

async def main():
    llm = ChatOpenAI(model="gpt-4o")

    structured_llm = llm.with_structured_output(Person)
    person = await structured_llm.invoke([
        UserMessage("Extract: John is 32 and works as a data scientist")
    ])

    print(f"{person.name}, {person.age}, {person.occupation}")
    # Output: John, 32, data scientist

asyncio.run(main())
```

### Tool Calling

Define tools using simple Python functions with the `@tool` decorator:
```python
from llmify import ChatOpenAI, UserMessage, ToolResultMessage, tool

@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get current weather for a location"""
    return f"Weather in {location}: 22°{unit[0].upper()}, Sunny"

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web"""
    return f"Found {max_results} results for '{query}'"

async def main():
    llm = ChatOpenAI(model="gpt-4o")
    tools = [get_weather, search_web]

    # Initial request
    messages = [UserMessage("What's the weather in Paris?")]
    response = await llm.invoke(messages, tools=tools)

    # Handle tool calls
    if response.has_tool_calls:
        messages.append(response.to_message())

        for tool_call in response.tool_calls:
            # Execute the tool
            result = tool_call.execute()

            # Add result to conversation
            messages.append(ToolResultMessage(
                tool_call_id=tool_call.id,
                content=result
            ))

        # Get final response
        final = await llm.invoke(messages, tools=tools)
        print(final.content)

asyncio.run(main())
```

**Key Points:**
- Type hints are automatically converted to JSON schema
- Tools are just decorated Python functions
- Built-in tool execution with `.execute()`

### Streaming

Stream responses token-by-token as they're generated:
```python
async def main():
    llm = ChatOpenAI(model="gpt-4o")

    async for chunk in llm.stream([
        UserMessage("Write a haiku about Python")
    ]):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

### Image Analysis

Analyze images using vision models:
```python
import base64
from llmify import ChatOpenAI, ImageMessage

async def main():
    llm = ChatOpenAI(model="gpt-4o")

    # Load and encode image
    with open("photo.jpg", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    response = await llm.invoke([
        ImageMessage(
            base64_data=image_data,
            media_type="image/jpeg",
            text="What's in this image?"
        )
    ])

    print(response)

asyncio.run(main())
```

## Configuration

### Environment Variables
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://.openai.azure.com/"
```

### Model Parameters

Set defaults when initializing or override per request:
```python
# Set defaults
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000
)

# Override per request
response = await llm.invoke(
    messages=[UserMessage("Hi")],
    temperature=0.2,  # More deterministic
    max_tokens=500
)
```

**Supported Parameters:**
- `temperature` - Creativity (0-2)
- `max_tokens` - Maximum response length
- `top_p` - Nucleus sampling
- `frequency_penalty` - Reduce repetition
- `presence_penalty` - Encourage diversity
- `stop` - Stop sequences
- `seed` - Deterministic outputs

## Providers

### OpenAI
```python
from llmify import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    api_key="sk-..."  # Optional if using env var
)
```

### Azure OpenAI
```python
from llmify import ChatAzureOpenAI

llm = ChatAzureOpenAI(
    model="gpt-4o",
    api_key="...",  # Optional if using env var
    azure_endpoint="https://.openai.azure.com/"  # Optional if using env var
)
```

## Design Philosophy

**LangChain-Inspired, but Simpler**
- Familiar message API (`SystemMessage`, `UserMessage`)
- Same interface across providers
- Less opinionated, more flexible

**Lightweight & Focused**
- Thin wrapper around official SDKs
- Minimal dependencies
- No unnecessary abstractions

**Type-Safe & Modern**
- Full type hints for IDE support
- Pydantic for validation
- Async-first design

## License

MIT
