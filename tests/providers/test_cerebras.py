from unittest.mock import patch

from llmify import CerebrasModel, ChatCerebras


class TestChatCerebras:
    @patch("llmify.providers.cerebras.AsyncOpenAI")
    def test_uses_cerebras_defaults(self, mock_client) -> None:
        model = ChatCerebras(api_key="test-key")

        assert model.model == "gpt-oss-120b"
        mock_client.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.cerebras.ai/v1",
            timeout=60.0,
            max_retries=2,
            default_headers=None,
        )

    @patch("llmify.providers.cerebras.AsyncOpenAI")
    def test_reads_api_key_from_environment(self, mock_client, monkeypatch) -> None:
        monkeypatch.setenv("CEREBRAS_API_KEY", "environment-key")

        ChatCerebras(model=CerebrasModel.GEMMA_4_31B_PREVIEW)

        assert mock_client.call_args.kwargs["api_key"] == "environment-key"

    @patch("llmify.providers.cerebras.AsyncOpenAI")
    def test_explicit_api_key_takes_precedence(self, mock_client, monkeypatch) -> None:
        monkeypatch.setenv("CEREBRAS_API_KEY", "environment-key")

        model = ChatCerebras(
            model="custom-dedicated-endpoint",
            api_key="explicit-key",
            default_headers={"X-Cerebras-Version-Patch": "2"},
        )

        assert model.model == "custom-dedicated-endpoint"
        assert mock_client.call_args.kwargs["api_key"] == "explicit-key"
        assert mock_client.call_args.kwargs["default_headers"] == {
            "X-Cerebras-Version-Patch": "2"
        }
