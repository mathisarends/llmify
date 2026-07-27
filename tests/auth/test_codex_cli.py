import asyncio
import base64
import io
import json
import time
import urllib.error
from pathlib import Path
from unittest.mock import patch

import pytest

from llmify.auth import (
    CodexCliAuth,
    CodexCredentials,
    CodexCredentialsError,
    codex_auth_path,
    codex_home,
    load_codex_credentials,
)
from llmify.auth.codex_cli import CodexRefreshResponse
from llmify.exceptions import CredentialsUnavailableError


def _jwt(expires_in: float) -> str:
    claims = json.dumps({"exp": time.time() + expires_in}).encode()
    payload = base64.urlsafe_b64encode(claims).decode().rstrip("=")
    return f"header.{payload}.signature"


def _write_auth(path: Path, tokens: dict | None = None, **overrides) -> Path:
    data = {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": _jwt(3600),
            "refresh_token": "refresh-1",
            "id_token": "id-1",
            "account_id": "acct-123",
        }
        | (tokens or {}),
    } | overrides
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _refreshed(**overrides) -> CodexRefreshResponse:
    return CodexRefreshResponse(
        access_token=_jwt(3600),
        refresh_token="refresh-2",
        id_token="id-2",
        **overrides,
    )


@pytest.fixture(autouse=True)
def _isolate_codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test away from the developer's real ~/.codex."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))


@pytest.fixture
def auth_file(tmp_path: Path) -> Path:
    return _write_auth(tmp_path / "auth.json")


class TestCodexHome:
    def test_uses_codex_home_environment_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert codex_home() == tmp_path
        assert codex_auth_path() == tmp_path / "auth.json"

    def test_falls_back_to_home_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CODEX_HOME", raising=False)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

        assert codex_home() == tmp_path / ".codex"


class TestCodexCredentials:
    def test_reports_seconds_until_expiry(self) -> None:
        credentials = CodexCredentials(
            access_token="token",
            account_id="acct-123",
            auth_path=Path("auth.json"),
            expires_at=time.time() + 600,
            refreshed=False,
        )

        assert credentials.expires_in == pytest.approx(600, abs=5)
        assert credentials.is_fresh

    def test_token_inside_the_skew_window_is_not_fresh(self) -> None:
        credentials = CodexCredentials(
            access_token="token",
            account_id=None,
            auth_path=Path("auth.json"),
            expires_at=time.time() + 30,
            refreshed=False,
        )

        assert not credentials.is_fresh

    def test_unknown_expiry_counts_as_fresh(self) -> None:
        credentials = CodexCredentials(
            access_token="token",
            account_id=None,
            auth_path=Path("auth.json"),
            expires_at=None,
            refreshed=False,
        )

        assert credentials.expires_in is None
        assert credentials.is_fresh


class TestLoadCodexCredentials:
    def test_returns_fresh_token_without_refreshing(self, auth_file: Path) -> None:
        with patch("llmify.auth.codex_cli._request_refresh") as request_refresh:
            credentials = load_codex_credentials(auth_path=auth_file)

        request_refresh.assert_not_called()
        assert credentials.account_id == "acct-123"
        assert credentials.auth_path == auth_file
        assert credentials.refreshed is False

    def test_reads_expiry_from_the_access_token(self, auth_file: Path) -> None:
        credentials = load_codex_credentials(auth_path=auth_file)

        assert credentials.expires_in == pytest.approx(3600, abs=5)

    def test_opaque_token_is_used_as_is(self, tmp_path: Path) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": "opaque"})

        with patch("llmify.auth.codex_cli._request_refresh") as request_refresh:
            credentials = load_codex_credentials(auth_path=path)

        request_refresh.assert_not_called()
        assert credentials.expires_at is None
        assert credentials.access_token == "opaque"

    def test_refreshes_a_token_that_is_about_to_expire(self, tmp_path: Path) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(30)})
        refreshed = _refreshed()

        with patch(
            "llmify.auth.codex_cli._request_refresh", return_value=refreshed
        ) as request_refresh:
            credentials = load_codex_credentials(auth_path=path)

        request_refresh.assert_called_once_with("refresh-1")
        assert credentials.refreshed is True
        assert credentials.access_token == refreshed.access_token
        assert credentials.account_id == "acct-123"

    def test_writes_rotated_tokens_back_to_disk(self, tmp_path: Path) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(-10)})
        refreshed = _refreshed()

        with patch("llmify.auth.codex_cli._request_refresh", return_value=refreshed):
            load_codex_credentials(auth_path=path)

        tokens = json.loads(path.read_text())["tokens"]
        assert tokens["access_token"] == refreshed.access_token
        assert tokens["refresh_token"] == "refresh-2"
        assert tokens["id_token"] == "id-2"
        assert json.loads(path.read_text())["last_refresh"]

    def test_preserves_keys_it_does_not_know(self, tmp_path: Path) -> None:
        path = _write_auth(
            tmp_path / "auth.json",
            {"access_token": _jwt(-10), "unknown_token_key": "keep-me"},
            OPENAI_API_KEY=None,
            unknown_top_level_key={"nested": 1},
        )

        with patch("llmify.auth.codex_cli._request_refresh", return_value=_refreshed()):
            load_codex_credentials(auth_path=path)

        data = json.loads(path.read_text())
        assert data["unknown_top_level_key"] == {"nested": 1}
        assert data["OPENAI_API_KEY"] is None
        assert data["tokens"]["unknown_token_key"] == "keep-me"

    def test_leaves_no_temporary_file_behind(self, tmp_path: Path) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(-10)})

        with patch("llmify.auth.codex_cli._request_refresh", return_value=_refreshed()):
            load_codex_credentials(auth_path=path)

        assert [entry.name for entry in tmp_path.iterdir()] == ["auth.json"]

    def test_keeps_the_old_token_when_the_refresh_omits_one(
        self, tmp_path: Path
    ) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(-10)})
        partial = CodexRefreshResponse(access_token=_jwt(3600))

        with patch("llmify.auth.codex_cli._request_refresh", return_value=partial):
            load_codex_credentials(auth_path=path)

        assert json.loads(path.read_text())["tokens"]["refresh_token"] == "refresh-1"

    def test_expired_token_is_returned_as_is_without_refresh(
        self, tmp_path: Path
    ) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(-10)})
        before = path.read_text()

        with patch("llmify.auth.codex_cli._request_refresh") as request_refresh:
            credentials = load_codex_credentials(auth_path=path, allow_refresh=False)

        request_refresh.assert_not_called()
        assert credentials.is_fresh is False
        assert path.read_text() == before


class TestLoadCodexCredentialsErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CodexCredentialsError, match="codex login"):
            load_codex_credentials(auth_path=tmp_path / "auth.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(CodexCredentialsError, match="not a usable auth file"):
            load_codex_credentials(auth_path=path)

    def test_api_key_login(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text(json.dumps({"auth_mode": "apikey"}), encoding="utf-8")

        with pytest.raises(CodexCredentialsError, match="Only ChatGPT OAuth logins"):
            load_codex_credentials(auth_path=path)

    def test_without_tokens(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text(json.dumps({"auth_mode": "chatgpt"}), encoding="utf-8")

        with pytest.raises(CodexCredentialsError, match="No ChatGPT tokens"):
            load_codex_credentials(auth_path=path)

    def test_expired_without_refresh_token(self, tmp_path: Path) -> None:
        path = _write_auth(
            tmp_path / "auth.json",
            {"access_token": _jwt(-10), "refresh_token": None},
        )

        with pytest.raises(CodexCredentialsError, match="no refresh token"):
            load_codex_credentials(auth_path=path)

    def test_is_a_credentials_error(self) -> None:
        assert issubclass(CodexCredentialsError, CredentialsUnavailableError)


class TestRefreshRequestErrors:
    @pytest.fixture
    def stale_auth_file(self, tmp_path: Path) -> Path:
        return _write_auth(tmp_path / "auth.json", {"access_token": _jwt(-10)})

    def _http_error(self, status: int, body: dict) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(
            url="https://auth.openai.com/oauth/token",
            code=status,
            msg="error",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(json.dumps(body).encode()),
        )

    @pytest.mark.parametrize(
        "code",
        ["refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated"],
    )
    def test_rejected_refresh_token_asks_for_a_new_login(
        self, stale_auth_file: Path, code: str
    ) -> None:
        error = self._http_error(400, {"error": code})

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(CodexCredentialsError, match=f"{code}.*codex login"):
                load_codex_credentials(auth_path=stale_auth_file)

    def test_other_http_error_reports_the_status(self, stale_auth_file: Path) -> None:
        error = self._http_error(500, {"error": "server_error"})

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(CodexCredentialsError, match="HTTP 500"):
                load_codex_credentials(auth_path=stale_auth_file)

    def test_network_error(self, stale_auth_file: Path) -> None:
        error = urllib.error.URLError("connection refused")

        with patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(CodexCredentialsError, match="network error"):
                load_codex_credentials(auth_path=stale_auth_file)

    def test_unreadable_refresh_response(self, stale_auth_file: Path) -> None:
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = b"nonsense"

            with pytest.raises(CodexCredentialsError, match="unexpected response"):
                load_codex_credentials(auth_path=stale_auth_file)


class TestCodexCliAuth:
    def test_exposes_the_login_it_reads(self, auth_file: Path) -> None:
        auth = CodexCliAuth(auth_path=auth_file)

        assert auth.account_id == "acct-123"
        assert auth.auth_path == auth_file
        assert auth.credentials.is_fresh

    def test_requires_an_account_id(self, tmp_path: Path) -> None:
        path = _write_auth(tmp_path / "auth.json", {"account_id": None})

        with pytest.raises(CodexCredentialsError, match="No ChatGPT account id"):
            CodexCliAuth(auth_path=path)

    def test_reads_the_login_eagerly(self, tmp_path: Path) -> None:
        with pytest.raises(CodexCredentialsError, match="codex login"):
            CodexCliAuth(auth_path=tmp_path / "auth.json")

    def test_does_not_refresh_an_expired_token_on_construction(
        self, tmp_path: Path
    ) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(-10)})

        with patch("llmify.auth.codex_cli._request_refresh") as request_refresh:
            auth = CodexCliAuth(auth_path=path)

        request_refresh.assert_not_called()
        assert not auth.credentials.is_fresh

    @pytest.mark.asyncio
    async def test_hands_out_the_cached_token_while_it_is_fresh(
        self, auth_file: Path
    ) -> None:
        auth = CodexCliAuth(auth_path=auth_file)

        with patch("llmify.auth.codex_cli.load_codex_credentials") as load:
            assert await auth() == auth.credentials.access_token
            assert await auth() == auth.credentials.access_token

        load.assert_not_called()

    @pytest.mark.asyncio
    async def test_refreshes_a_token_that_is_about_to_expire(
        self, tmp_path: Path
    ) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(30)})
        refreshed = _refreshed()

        with patch("llmify.auth.codex_cli._request_refresh", return_value=refreshed):
            auth = CodexCliAuth(auth_path=path)
            token = await auth()

        assert token == refreshed.access_token
        assert auth.credentials.refreshed is True

    @pytest.mark.asyncio
    async def test_concurrent_requests_trigger_a_single_refresh(
        self, tmp_path: Path
    ) -> None:
        # The OAuth server rotates refresh tokens, so a second concurrent
        # refresh would be rejected as reused.
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(30)})
        refreshed = _refreshed()

        def slow_refresh(refresh_token: str) -> CodexRefreshResponse:
            time.sleep(0.01)
            return refreshed

        with patch(
            "llmify.auth.codex_cli._request_refresh", side_effect=slow_refresh
        ) as request_refresh:
            auth = CodexCliAuth(auth_path=path)
            tokens = await asyncio.gather(*(auth() for _ in range(8)))

        request_refresh.assert_called_once()
        assert set(tokens) == {refreshed.access_token}
