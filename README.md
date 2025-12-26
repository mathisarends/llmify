# llmify

A lightweight wrapper for LLM providers (OpenAI, Azure, Anthropic, Google Gemini) with a clean, LangChain-inspired API.

## Installation

```bash
pip install py-llmify
```

## Quick Start

### Basic Completion

```python
import asyncio
from llmify import ChatOpenAI, UserMessage, SystemMessage

async def main():
    llm = ChatOpenAI(model="gpt-4o")

    response = await llm.invoke([
        SystemMessage("You are a helpful assistant"),
        UserMessage("What is 2+2?")
    ])
    print(response)

asyncio.run(main())
```

### Structured Output

Generate structured data with Pydantic models:

```python
from pydantic import BaseModel
from llmify import ChatOpenAI, UserMessage

class Person(BaseModel):
    name: str
    age: int
    occupation: str

async def main():
    llm = ChatOpenAI()

    person = await llm.invoke_structured([
        UserMessage("Extract: Anna is 28 and works as a Software Engineer")
    ], response_model=Person)

    print(f"Name: {person.name}, Age: {person.age}")

asyncio.run(main())
```

### Streaming

Stream tokens as they are generated:

```python
async def main():
    llm = ChatOpenAI()

    print("Response: ", end="", flush=True)
    async for chunk in llm.stream([
        UserMessage("Tell me a short story")
    ]):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

### Image Analysis

```python
import base64
from llmify import ChatOpenAI, UserMessage, ImageMessage

async def main():
    llm = ChatOpenAI()

    # Load and encode image
    with open("photo.jpg", "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')

    response = await llm.invoke([
        UserMessage("Describe this image:"),
        ImageMessage(
            base64_data=base64_image,
            media_type="image/jpeg"
        )
    ])
    print(response)

asyncio.run(main())
```

## Configuration

### API Keys

Set environment variables or pass directly:

```python
# Environment variable
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

llm = ChatOpenAI(api_key="sk-...")
```

### Generation Parameters

Control model behavior with defaults or per-call overrides:

```python
# Set defaults
llm = ChatOpenAI(
    model="gpt-4o",
    max_tokens=1000,
    temperature=0.7,
    top_p=0.9
)

# Override per call
response = await llm.invoke(
    [UserMessage("Hi")],
    temperature=1.0,  # Override
    max_tokens=2000
)
```

## Supported Parameters

### Common Parameters
- `max_tokens` - Maximum tokens in response
- `temperature` - Creativity (0-2)
- `top_p` - Nucleus sampling
- `stop` - Stop sequences

### OpenAI/Azure Specific
- `frequency_penalty` - Reduce repetition
- `presence_penalty` - Encourage new topics
- `seed` - Deterministic outputs
- `response_format` - JSON mode

### Client Config
- `timeout` - Request timeout
- `max_retries` - Retry attempts

## Message Types

### UserMessage
```python
UserMessage("What is AI?")
```

### SystemMessage
```python
SystemMessage("You are an expert in quantum physics")
```

### ImageMessage
```python
ImageMessage(
    base64_data="iVBORw0KGgo...",
    media_type="image/png",  # or "image/jpeg"
    text="Describe this"
)
```

## Providers

### OpenAI

```python
from llmify import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    api_key="sk-...",  # or env var OPENAI_API_KEY
)
```

### Azure OpenAI

```python
from llmify import ChatAzureOpenAI

llm = ChatAzureOpenAI(
    model="gpt-4o",
    api_key="...",  # or env var AZURE_OPENAI_API_KEY
    azure_endpoint="https://<name>.openai.azure.com/",  # or env var AZURE_OPENAI_ENDPOINT
)
```

## Design Philosophy

**Lightweight & Simple**: Thin wrapper around official SDKs, not a heavy framework.

**LangChain-Inspired**: Familiar message API (`SystemMessage`, `UserMessage`, `ImageMessage`).

**Unified Interface**: Same API across all providers - swap providers with one line.

**Parameter Control**: Sensible defaults + per-call overrides (like official SDKs).

## License

MIT
