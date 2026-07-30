from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("openai")

from llmify.exceptions import (
    AuthenticationError,
    CredentialsUnavailableError,
    LLMifyError,
)
from llmify.messages import UserMessage
from llmify.providers.azure import ChatAzureOpenAI
from llmify.providers.openai import ChatOpenAI


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


class TestClientConfiguration:
    def test_llmify_owns_the_retry_budget(self) -> None:
        llm = ChatOpenAI(max_retries=4)

        assert llm._default_max_retries == 4
        assert llm._client.max_retries == 0

    def test_base_url_is_forwarded(self) -> None:
        llm = ChatOpenAI(base_url="https://chatgpt.com/backend-api/codex")

        assert str(llm._client.base_url) == "https://chatgpt.com/backend-api/codex/"

    def test_base_url_defaults_to_the_openai_api(self) -> None:
        llm = ChatOpenAI()

        assert str(llm._client.base_url) == "https://api.openai.com/v1/"

    def test_default_headers_are_forwarded(self) -> None:
        llm = ChatOpenAI(default_headers={"ChatGPT-Account-Id": "acct-123"})

        assert llm._client.default_headers["ChatGPT-Account-Id"] == "acct-123"

    def test_api_key_from_environment(self) -> None:
        assert ChatOpenAI()._client.api_key == "sk-test"

    def test_explicit_api_key_wins_over_environment(self) -> None:
        assert ChatOpenAI(api_key="sk-explicit")._client.api_key == "sk-explicit"

    @pytest.mark.asyncio
    async def test_callable_api_key_is_resolved_per_request(self) -> None:
        tokens = iter(["tok-first", "tok-second"])

        async def next_token() -> str:
            return next(tokens)

        client = ChatOpenAI(api_key=next_token)._client

        await client._refresh_api_key()
        assert client.auth_headers["Authorization"] == "Bearer tok-first"

        await client._refresh_api_key()
        assert client.auth_headers["Authorization"] == "Bearer tok-second"

    @patch("llmify.providers.azure.AsyncAzureOpenAI")
    def test_azure_disables_sdk_retries(self, client) -> None:
        ChatAzureOpenAI(
            api_key="test-key",
            azure_endpoint="https://example.openai.azure.com",
            max_retries=4,
        )

        assert client.call_args.kwargs["max_retries"] == 0


class TestCredentialsUnavailableError:
    def test_is_catchable_as_authentication_error(self) -> None:
        error = CredentialsUnavailableError("codex session expired, run 'codex login'")

        assert isinstance(error, AuthenticationError)
        assert isinstance(error, LLMifyError)
        assert str(error) == "codex session expired, run 'codex login'"

    @pytest.mark.asyncio
    async def test_survives_the_request_path_unmapped(self) -> None:
        llm = ChatOpenAI()
        llm._client.chat.completions.create = AsyncMock(
            side_effect=CredentialsUnavailableError("token refresh failed")
        )

        with pytest.raises(CredentialsUnavailableError, match="token refresh failed"):
            await llm.invoke([UserMessage(content="hi")])
