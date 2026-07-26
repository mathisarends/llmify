from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from llmify import ChatCodex
from llmify.exceptions import CredentialsUnavailableError


class TestChatCodex:
    @patch("llmify.providers.openai_responses.AsyncOpenAI")
    def test_configures_codex_endpoint_and_account_header(self, mock_client) -> None:
        ChatCodex(
            model="gpt-test",
            api_key="access-token",
            chatgpt_account_id="account-123",
        )

        mock_client.assert_called_once_with(
            api_key="access-token",
            base_url="https://chatgpt.com/backend-api/codex",
            timeout=60.0,
            max_retries=2,
            default_headers={"ChatGPT-Account-Id": "account-123"},
        )

    @patch("llmify.providers.openai_responses.AsyncOpenAI")
    def test_reads_access_key_from_codex_environment(
        self, mock_client, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CODEX_ACCESS_KEY", "environment-token")

        ChatCodex(model="gpt-test", chatgpt_account_id="account-123")

        assert mock_client.call_args.kwargs["api_key"] == "environment-token"

    @patch("llmify.providers.openai_responses.AsyncOpenAI")
    def test_preserves_additional_headers(self, mock_client) -> None:
        ChatCodex(
            model="gpt-test",
            api_key="access-token",
            chatgpt_account_id="account-123",
            default_headers={"X-Trace-Id": "trace-123"},
        )

        assert mock_client.call_args.kwargs["default_headers"] == {
            "X-Trace-Id": "trace-123",
            "ChatGPT-Account-Id": "account-123",
        }

    def test_rejects_empty_account_id(self) -> None:
        with pytest.raises(ValueError, match="chatgpt_account_id"):
            ChatCodex(
                model="gpt-test",
                api_key="access-token",
                chatgpt_account_id="",
            )

    def test_requires_codex_access_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CODEX_ACCESS_KEY", raising=False)

        with pytest.raises(CredentialsUnavailableError, match="CODEX_ACCESS_KEY"):
            ChatCodex(model="gpt-test", chatgpt_account_id="account-123")
