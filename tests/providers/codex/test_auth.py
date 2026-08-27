import asyncio
import base64
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llmify.providers.codex import (
    CodexCliAuth,
    CodexCredentials,
    CodexCredentialsError,
    codex_auth_path,
    codex_home,
    read_codex_credentials,
    refresh_codex_credentials,
)
from llmify.providers.codex.auth import CodexRefreshResponse
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


def _http_response(status: int, body: dict | str) -> httpx.Response:
    request = httpx.Request("POST", "https://auth.openai.com/oauth/token")
    if isinstance(body, str):
        return httpx.Response(status, text=body, request=request)
    return httpx.Response(status, json=body, request=request)


@pytest.fixture(autouse=True)
def _isolate_codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test away from the developer's real ~/.codex."""
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))


@pytest.fixture
def auth_file(tmp_path: Path) -> Path:
    return _write_auth(tmp_path / "auth.json")


@pytest.fixture
def stale_auth_file(tmp_path: Path) -> Path:
    return _write_auth(tmp_path / "auth.json", {"access_token": _jwt(-10)})


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


class TestReadCodexCredentials:
    def test_returns_the_stored_login(self, auth_file: Path) -> None:
        credentials = read_codex_credentials(auth_path=auth_file)

        assert credentials.account_id == "acct-123"
        assert credentials.auth_path == auth_file
        assert credentials.refreshed is False
        assert credentials.expires_in == pytest.approx(3600, abs=5)

    def test_returns_an_expired_token_as_is(self, stale_auth_file: Path) -> None:
        before = stale_auth_file.read_text()

        credentials = read_codex_credentials(auth_path=stale_auth_file)

        assert credentials.is_fresh is False
        assert stale_auth_file.read_text() == before

    def test_opaque_token_has_no_expiry(self, tmp_path: Path) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": "opaque"})

        credentials = read_codex_credentials(auth_path=path)

        assert credentials.expires_at is None
        assert credentials.access_token == "opaque"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CodexCredentialsError, match="codex login"):
            read_codex_credentials(auth_path=tmp_path / "auth.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(CodexCredentialsError, match="not a usable auth file"):
            read_codex_credentials(auth_path=path)

    def test_api_key_login(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text(json.dumps({"auth_mode": "apikey"}), encoding="utf-8")

        with pytest.raises(CodexCredentialsError, match="Only ChatGPT OAuth logins"):
            read_codex_credentials(auth_path=path)

    def test_without_tokens(self, tmp_path: Path) -> None:
        path = tmp_path / "auth.json"
        path.write_text(json.dumps({"auth_mode": "chatgpt"}), encoding="utf-8")

        with pytest.raises(CodexCredentialsError, match="No ChatGPT tokens"):
            read_codex_credentials(auth_path=path)

    def test_is_a_credentials_error(self) -> None:
        assert issubclass(CodexCredentialsError, CredentialsUnavailableError)


@pytest.mark.asyncio
class TestRefreshCodexCredentials:
    async def test_leaves_a_fresh_token_alone(self, auth_file: Path) -> None:
        with patch("llmify.providers.codex.auth._request_refresh") as request_refresh:
            credentials = await refresh_codex_credentials(auth_path=auth_file)

        request_refresh.assert_not_called()
        assert credentials.refreshed is False

    async def test_refreshes_a_token_that_is_about_to_expire(
        self, tmp_path: Path
    ) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(30)})
        refreshed = _refreshed()

        with patch(
            "llmify.providers.codex.auth._request_refresh",
            AsyncMock(return_value=refreshed),
        ) as request_refresh:
            credentials = await refresh_codex_credentials(auth_path=path)

        request_refresh.assert_awaited_once_with("refresh-1")
        assert credentials.refreshed is True
        assert credentials.access_token == refreshed.access_token
        assert credentials.account_id == "acct-123"

    async def test_adopts_a_token_another_process_refreshed(
        self, stale_auth_file: Path
    ) -> None:
        # The Codex CLI may have refreshed since we last read the file; using
        # its token avoids rotating the refresh token a second time.
        _write_auth(stale_auth_file, {"access_token": _jwt(3600)})

        with patch("llmify.providers.codex.auth._request_refresh") as request_refresh:
            credentials = await refresh_codex_credentials(auth_path=stale_auth_file)

        request_refresh.assert_not_called()
        assert credentials.is_fresh

    async def test_writes_rotated_tokens_back_to_disk(
        self, stale_auth_file: Path
    ) -> None:
        refreshed = _refreshed()

        with patch(
            "llmify.providers.codex.auth._request_refresh",
            AsyncMock(return_value=refreshed),
        ):
            await refresh_codex_credentials(auth_path=stale_auth_file)

        data = json.loads(stale_auth_file.read_text())
        assert data["tokens"]["access_token"] == refreshed.access_token
        assert data["tokens"]["refresh_token"] == "refresh-2"
        assert data["tokens"]["id_token"] == "id-2"
        assert data["last_refresh"]

    async def test_preserves_keys_it_does_not_know(self, tmp_path: Path) -> None:
        path = _write_auth(
            tmp_path / "auth.json",
            {"access_token": _jwt(-10), "unknown_token_key": "keep-me"},
            OPENAI_API_KEY=None,
            unknown_top_level_key={"nested": 1},
        )

        with patch(
            "llmify.providers.codex.auth._request_refresh",
            AsyncMock(return_value=_refreshed()),
        ):
            await refresh_codex_credentials(auth_path=path)

        data = json.loads(path.read_text())
        assert data["unknown_top_level_key"] == {"nested": 1}
        assert data["OPENAI_API_KEY"] is None
        assert data["tokens"]["unknown_token_key"] == "keep-me"

    async def test_leaves_no_temporary_file_behind(
        self, tmp_path: Path, stale_auth_file: Path
    ) -> None:
        with patch(
            "llmify.providers.codex.auth._request_refresh",
            AsyncMock(return_value=_refreshed()),
        ):
            await refresh_codex_credentials(auth_path=stale_auth_file)

        assert [entry.name for entry in tmp_path.iterdir()] == ["auth.json"]

    async def test_keeps_the_old_token_when_the_refresh_omits_one(
        self, stale_auth_file: Path
    ) -> None:
        partial = CodexRefreshResponse(access_token=_jwt(3600))

        with patch(
            "llmify.providers.codex.auth._request_refresh",
            AsyncMock(return_value=partial),
        ):
            await refresh_codex_credentials(auth_path=stale_auth_file)

        tokens = json.loads(stale_auth_file.read_text())["tokens"]
        assert tokens["refresh_token"] == "refresh-1"

    async def test_expired_without_refresh_token(self, tmp_path: Path) -> None:
        path = _write_auth(
            tmp_path / "auth.json",
            {"access_token": _jwt(-10), "refresh_token": None},
        )

        with pytest.raises(CodexCredentialsError, match="no refresh token"):
            await refresh_codex_credentials(auth_path=path)


@pytest.mark.asyncio
class TestRefreshRequestErrors:
    @pytest.mark.parametrize(
        "code",
        ["refresh_token_expired", "refresh_token_reused", "refresh_token_invalidated"],
    )
    async def test_rejected_refresh_token_asks_for_a_new_login(
        self, stale_auth_file: Path, code: str
    ) -> None:
        response = _http_response(400, {"error": code})

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)):
            with pytest.raises(CodexCredentialsError, match=f"{code}.*codex login"):
                await refresh_codex_credentials(auth_path=stale_auth_file)

    async def test_other_http_error_reports_the_status(
        self, stale_auth_file: Path
    ) -> None:
        response = _http_response(500, {"error": "server_error"})

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)):
            with pytest.raises(CodexCredentialsError, match="HTTP 500"):
                await refresh_codex_credentials(auth_path=stale_auth_file)

    async def test_network_error(self, stale_auth_file: Path) -> None:
        error = httpx.ConnectError("connection refused")

        with patch("httpx.AsyncClient.post", AsyncMock(side_effect=error)):
            with pytest.raises(CodexCredentialsError, match="network error"):
                await refresh_codex_credentials(auth_path=stale_auth_file)

    async def test_unreadable_refresh_response(self, stale_auth_file: Path) -> None:
        response = _http_response(200, "nonsense")

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)):
            with pytest.raises(CodexCredentialsError, match="unexpected response"):
                await refresh_codex_credentials(auth_path=stale_auth_file)

    async def test_does_not_touch_the_file_when_the_refresh_fails(
        self, stale_auth_file: Path
    ) -> None:
        before = stale_auth_file.read_text()
        response = _http_response(500, {"error": "server_error"})

        with patch("httpx.AsyncClient.post", AsyncMock(return_value=response)):
            with pytest.raises(CodexCredentialsError):
                await refresh_codex_credentials(auth_path=stale_auth_file)

        assert stale_auth_file.read_text() == before


class TestCodexCliAuth:
    def test_exposes_the_login_it_reads(self, auth_file: Path) -> None:
        auth = CodexCliAuth(read_codex_credentials(auth_path=auth_file))

        assert auth.account_id == "acct-123"
        assert auth.auth_path == auth_file
        assert auth.credentials.is_fresh

    def test_requires_an_account_id(self, tmp_path: Path) -> None:
        path = _write_auth(tmp_path / "auth.json", {"account_id": None})

        with pytest.raises(CodexCredentialsError, match="No ChatGPT account id"):
            CodexCliAuth(read_codex_credentials(auth_path=path))

    def test_accepts_an_expired_login_without_touching_it(
        self, stale_auth_file: Path
    ) -> None:
        before = stale_auth_file.read_text()

        with patch("httpx.AsyncClient.post") as post:
            auth = CodexCliAuth(read_codex_credentials(auth_path=stale_auth_file))

        post.assert_not_called()
        assert not auth.credentials.is_fresh
        assert stale_auth_file.read_text() == before

    @pytest.mark.asyncio
    async def test_hands_out_the_cached_token_while_it_is_fresh(
        self, auth_file: Path
    ) -> None:
        auth = CodexCliAuth(read_codex_credentials(auth_path=auth_file))

        with patch("llmify.providers.codex.auth.refresh_codex_credentials") as refresh:
            assert await auth() == auth.credentials.access_token
            assert await auth() == auth.credentials.access_token

        refresh.assert_not_called()

    @pytest.mark.asyncio
    async def test_refreshes_a_token_that_is_about_to_expire(
        self, tmp_path: Path
    ) -> None:
        path = _write_auth(tmp_path / "auth.json", {"access_token": _jwt(30)})
        refreshed = _refreshed()

        with patch(
            "llmify.providers.codex.auth._request_refresh",
            AsyncMock(return_value=refreshed),
        ):
            auth = CodexCliAuth(read_codex_credentials(auth_path=path))
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

        async def slow_refresh(refresh_token: str) -> CodexRefreshResponse:
            await asyncio.sleep(0.01)
            return refreshed

        with patch(
            "llmify.providers.codex.auth._request_refresh", side_effect=slow_refresh
        ) as request_refresh:
            auth = CodexCliAuth(read_codex_credentials(auth_path=path))
            tokens = await asyncio.gather(*(auth() for _ in range(8)))

        request_refresh.assert_awaited_once()
        assert set(tokens) == {refreshed.access_token}
