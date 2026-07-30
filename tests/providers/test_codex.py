import base64
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("openai")

from llmify import ChatCodex
from llmify.auth import CodexCliAuth, CodexCredentialsError
from llmify.exceptions import CredentialsUnavailableError


def _auth_file(path: Path, account_id: str | None = "acct-123") -> Path:
    claims = json.dumps({"exp": time.time() + 3600}).encode()
    payload = base64.urlsafe_b64encode(claims).decode().rstrip("=")
    path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": f"header.{payload}.signature",
                    "refresh_token": "refresh-1",
                    "account_id": account_id,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


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


class TestChatCodexFromCli:
    @patch("llmify.providers.openai_responses.AsyncOpenAI")
    @pytest.mark.asyncio
    async def test_borrows_account_id_and_token_from_the_cli(
        self, mock_client, tmp_path: Path
    ) -> None:
        auth_path = _auth_file(tmp_path / "auth.json")

        ChatCodex.from_cli(model="gpt-test", auth_path=auth_path)

        stored_token = json.loads(auth_path.read_text())["tokens"]["access_token"]
        kwargs = mock_client.call_args.kwargs

        assert kwargs["default_headers"] == {"ChatGPT-Account-Id": "acct-123"}
        assert isinstance(kwargs["api_key"], CodexCliAuth)
        assert await kwargs["api_key"]() == stored_token

    @patch("llmify.providers.openai_responses.AsyncOpenAI")
    def test_forwards_model_options(self, mock_client, tmp_path: Path) -> None:
        llm = ChatCodex.from_cli(
            model="gpt-test",
            auth_path=_auth_file(tmp_path / "auth.json"),
            temperature=0.2,
            max_retries=5,
        )

        assert llm.model == "gpt-test"
        assert mock_client.call_args.kwargs["max_retries"] == 5

    @patch("llmify.providers.openai_responses.AsyncOpenAI")
    def test_forwards_the_reasoning_effort(self, mock_client, tmp_path: Path) -> None:
        llm = ChatCodex.from_cli(
            model="gpt-test",
            auth_path=_auth_file(tmp_path / "auth.json"),
            reasoning_effort="xhigh",
        )

        assert llm._default_kwargs["reasoning_effort"] == "xhigh"

    def test_reports_a_missing_login(self, tmp_path: Path) -> None:
        with pytest.raises(CodexCredentialsError, match="codex login"):
            ChatCodex.from_cli(model="gpt-test", auth_path=tmp_path / "missing.json")

    def test_reports_a_login_without_account_id(self, tmp_path: Path) -> None:
        auth_path = _auth_file(tmp_path / "auth.json", account_id=None)

        with pytest.raises(CodexCredentialsError, match="No ChatGPT account id"):
            ChatCodex.from_cli(model="gpt-test", auth_path=auth_path)
