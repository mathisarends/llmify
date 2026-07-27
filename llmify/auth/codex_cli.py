"""Borrow the ChatGPT OAuth credentials of the locally installed Codex CLI.

The idea, and the reverse-engineered refresh flow below (endpoint, client id,
`auth.json` layout), follow https://github.com/simonw/llm-openai-via-codex.
"""

import asyncio
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llmify.exceptions import CredentialsUnavailableError

_REFRESH_URL = "https://auth.openai.com/oauth/token"
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_REFRESH_SKEW_SECONDS = 120.0

_INVALID_REFRESH_TOKEN_CODES = frozenset(
    {
        "refresh_token_expired",
        "refresh_token_reused",
        "refresh_token_invalidated",
    }
)


class CodexCredentialsError(CredentialsUnavailableError):
    """Raised when the local Codex credentials are missing, unusable or unrefreshable."""


class CodexCredentials(BaseModel):
    model_config = ConfigDict(frozen=True)

    access_token: str
    account_id: str | None
    auth_path: Path
    expires_at: float | None
    refreshed: bool

    @property
    def expires_in(self) -> float | None:
        if self.expires_at is None:
            return None
        return self.expires_at - time.time()

    @property
    def is_fresh(self) -> bool:
        # An unreadable expiry counts as fresh, otherwise every single request
        # would rotate the CLI's refresh token.
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at - _REFRESH_SKEW_SECONDS


class CodexTokens(BaseModel):
    # The CLI owns this file, so unknown keys are carried through a refresh
    # instead of being dropped on write-back.
    model_config = ConfigDict(extra="allow")

    access_token: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None
    account_id: str | None = None


class CodexAuthFile(BaseModel):
    model_config = ConfigDict(extra="allow")

    auth_mode: str | None = None
    tokens: CodexTokens = Field(default_factory=CodexTokens)
    last_refresh: str | None = None


class CodexRefreshResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    access_token: str | None = None
    refresh_token: str | None = None
    id_token: str | None = None


class _RefreshError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: str | None = None


class _JwtClaims(BaseModel):
    model_config = ConfigDict(extra="ignore")

    exp: float | None = None


class CodexCliAuth:
    """Async `api_key` provider that keeps the Codex CLI's access token fresh."""

    def __init__(self, auth_path: Path | None = None) -> None:
        credentials = load_codex_credentials(auth_path=auth_path, allow_refresh=False)
        if not credentials.account_id:
            raise CodexCredentialsError(
                f"No ChatGPT account id found in {credentials.auth_path}. "
                "Run `codex login` first."
            )

        self._account_id = credentials.account_id
        self._credentials = credentials
        self._lock = asyncio.Lock()

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def auth_path(self) -> Path:
        return self._credentials.auth_path

    @property
    def credentials(self) -> CodexCredentials:
        return self._credentials

    async def __call__(self) -> str:
        # The lock keeps concurrent requests to a single refresh: the OAuth
        # server rotates refresh tokens, so a second one would be rejected as
        # reused and invalidate the login for the CLI too.
        async with self._lock:
            if not self._credentials.is_fresh:
                self._credentials = await asyncio.to_thread(
                    load_codex_credentials, auth_path=self._credentials.auth_path
                )
            return self._credentials.access_token


def codex_home() -> Path:
    codex_home_env = os.environ.get("CODEX_HOME")
    if codex_home_env:
        return Path(codex_home_env).expanduser()
    return Path.home() / ".codex"


def codex_auth_path() -> Path:
    return codex_home() / "auth.json"


def load_codex_credentials(
    *,
    auth_path: Path | None = None,
    allow_refresh: bool = True,
) -> CodexCredentials:
    """Read the Codex credentials, refreshing them when near expiry.

    Blocking: touches the filesystem and, when refreshing, the network.
    With `allow_refresh=False` an expired token is returned as-is instead.
    """
    path = auth_path or codex_auth_path()
    auth = _read_auth(path)
    tokens = auth.tokens

    if not tokens.access_token:
        raise CodexCredentialsError(
            f"No ChatGPT tokens found in {path}. Run `codex login` first."
        )

    credentials = CodexCredentials(
        access_token=tokens.access_token,
        account_id=tokens.account_id,
        auth_path=path,
        expires_at=_jwt_exp(tokens.access_token),
        refreshed=False,
    )

    if credentials.is_fresh or not allow_refresh:
        return credentials

    if not tokens.refresh_token:
        raise CodexCredentialsError(
            f"Access token in {path} is expired and no refresh token is available. "
            "Run `codex login` to re-authenticate."
        )

    refreshed = _request_refresh(tokens.refresh_token)
    tokens.access_token = refreshed.access_token or tokens.access_token
    tokens.refresh_token = refreshed.refresh_token or tokens.refresh_token
    tokens.id_token = refreshed.id_token or tokens.id_token
    auth.last_refresh = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _write_auth(path, auth)

    return CodexCredentials(
        access_token=tokens.access_token,
        account_id=tokens.account_id,
        auth_path=path,
        expires_at=_jwt_exp(tokens.access_token),
        refreshed=True,
    )


def _read_auth(path: Path) -> CodexAuthFile:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CodexCredentialsError(
            f"Codex auth file not found at {path}. Run `codex login` first."
        ) from None
    except OSError as exc:
        raise CodexCredentialsError(f"Cannot read {path}: {exc}") from None

    try:
        auth = CodexAuthFile.model_validate_json(raw)
    except ValidationError as exc:
        raise CodexCredentialsError(
            f"{path} is not a usable auth file: {exc}"
        ) from None

    if auth.auth_mode != "chatgpt":
        raise CodexCredentialsError(
            f"Expected auth_mode 'chatgpt' in {path}, got '{auth.auth_mode}'. "
            "Only ChatGPT OAuth logins are supported."
        )
    return auth


def _write_auth(path: Path, auth: CodexAuthFile) -> None:
    # `exclude_unset` keeps keys the CLI never wrote out of the file, so a
    # refresh only ever adds what it actually changed.
    payload = auth.model_dump_json(indent=2, exclude_unset=True)

    # The Codex CLI owns this file and may write it at the same time.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(payload, encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as exc:
        tmp_path.unlink(missing_ok=True)
        raise CodexCredentialsError(
            f"Cannot write refreshed tokens to {path}: {exc}"
        ) from None

    try:
        path.chmod(0o600)
    except OSError:
        pass


def _jwt_exp(token: str) -> float | None:
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        claims = _JwtClaims.model_validate_json(base64.urlsafe_b64decode(payload_b64))
    except (IndexError, ValueError, ValidationError):
        return None
    return claims.exp


def _request_refresh(refresh_token: str) -> CodexRefreshResponse:
    body = json.dumps(
        {
            "client_id": _CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode()

    request = urllib.request.Request(
        _REFRESH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request) as response:
            return CodexRefreshResponse.model_validate_json(response.read())
    except ValidationError as exc:
        raise CodexCredentialsError(
            f"Token refresh returned an unexpected response: {exc}"
        ) from None
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        error_code = _error_code(error_body)
        if error_code in _INVALID_REFRESH_TOKEN_CODES:
            raise CodexCredentialsError(
                f"Refresh token is no longer valid ({error_code}). "
                "Run `codex login` to re-authenticate."
            ) from None
        raise CodexCredentialsError(
            f"Token refresh failed (HTTP {exc.code}): {error_body}"
        ) from None
    except urllib.error.URLError as exc:
        raise CodexCredentialsError(
            f"Token refresh failed (network error): {exc.reason}"
        ) from None


def _error_code(error_body: str) -> str | None:
    try:
        return _RefreshError.model_validate_json(error_body).error
    except ValidationError:
        return None
