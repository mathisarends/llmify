"""Borrow the ChatGPT OAuth credentials of the locally installed Codex CLI.

Standalone: only the Python standard library, no `llm` / `openai` / `pydantic`.
The returned credentials also carry the path they were read from, so callers can
show the user where the tokens live on this machine.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REFRESH_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_SKEW_SECONDS = 30

_INVALID_REFRESH_TOKEN_CODES = frozenset(
    {
        "refresh_token_expired",
        "refresh_token_reused",
        "refresh_token_invalidated",
    }
)


class CodexCredentialsError(Exception):
    """Raised when the local Codex credentials are missing, unusable or unrefreshable."""


@dataclass(frozen=True, slots=True)
class CodexCredentials:
    """Credentials borrowed from the Codex CLI, plus where they came from."""

    access_token: str
    account_id: str | None
    auth_path: Path
    expires_at: float | None
    refreshed: bool

    @property
    def expires_in(self) -> float | None:
        """Seconds until the access token expires, or None if it carries no expiry."""
        if self.expires_at is None:
            return None
        return self.expires_at - time.time()


def codex_home() -> Path:
    """Directory the Codex CLI stores its state in (`$CODEX_HOME`, else `~/.codex`)."""
    codex_home_env = os.environ.get("CODEX_HOME")
    if codex_home_env:
        return Path(codex_home_env).expanduser()
    return Path.home() / ".codex"


def codex_auth_path() -> Path:
    """Path of the Codex `auth.json`, whether or not it exists yet."""
    return codex_home() / "auth.json"


def load_codex_credentials(
    *,
    auth_path: Path | None = None,
    allow_refresh: bool = True,
) -> CodexCredentials:
    """Read the Codex ChatGPT OAuth credentials, refreshing them when near expiry.

    Args:
        auth_path: Override for the `auth.json` location. Defaults to `codex_auth_path()`.
        allow_refresh: When False, expired tokens are returned as-is instead of
            being exchanged for fresh ones (no network access, no file writes).

    Raises:
        CodexCredentialsError: If the file is missing, not a ChatGPT OAuth login,
            holds no tokens, or the refresh was rejected.
    """
    path = auth_path or codex_auth_path()
    data = _read_auth(path)

    tokens = data.get("tokens") or {}
    access_token = tokens.get("access_token")
    if not access_token:
        raise CodexCredentialsError(
            f"No ChatGPT tokens found in {path}. Run `codex login` first."
        )

    account_id = tokens.get("account_id")
    expires_at = _jwt_exp(access_token)

    is_fresh = expires_at is not None and time.time() < (
        expires_at - REFRESH_SKEW_SECONDS
    )
    if is_fresh or not allow_refresh:
        return CodexCredentials(
            access_token=access_token,
            account_id=account_id,
            auth_path=path,
            expires_at=expires_at,
            refreshed=False,
        )

    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise CodexCredentialsError(
            f"Access token in {path} is expired and no refresh token is available. "
            "Run `codex login` to re-authenticate."
        )

    new_tokens = _request_refresh(refresh_token)
    for key in ("access_token", "id_token", "refresh_token"):
        if new_tokens.get(key):
            tokens[key] = new_tokens[key]

    data["tokens"] = tokens
    data["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    _write_auth(path, data)

    return CodexCredentials(
        access_token=tokens["access_token"],
        account_id=account_id,
        auth_path=path,
        expires_at=_jwt_exp(tokens["access_token"]),
        refreshed=True,
    )


def _read_auth(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise CodexCredentialsError(
            f"Codex auth file not found at {path}. Run `codex login` first."
        ) from None
    except OSError as exc:
        raise CodexCredentialsError(f"Cannot read {path}: {exc}") from None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CodexCredentialsError(f"{path} is not valid JSON: {exc}") from None

    auth_mode = data.get("auth_mode")
    if auth_mode != "chatgpt":
        raise CodexCredentialsError(
            f"Expected auth_mode 'chatgpt' in {path}, got '{auth_mode}'. "
            "Only ChatGPT OAuth logins are supported."
        )
    return data


def _write_auth(path: Path, data: dict[str, Any]) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    try:
        path.chmod(0o600)
    except OSError:
        # Best effort: POSIX permissions are largely meaningless on Windows.
        pass


def _jwt_exp(token: str) -> float | None:
    """Expiry claim of a JWT access token, or None if it cannot be read."""
    try:
        payload_b64 = token.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
    except (IndexError, ValueError, json.JSONDecodeError):
        return None
    exp = payload.get("exp")
    return float(exp) if isinstance(exp, (int, float)) else None


def _request_refresh(refresh_token: str) -> dict[str, Any]:
    body = json.dumps(
        {
            "client_id": CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
    ).encode()
    request = urllib.request.Request(
        REFRESH_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode(errors="replace")
        if _error_code(error_body) in _INVALID_REFRESH_TOKEN_CODES:
            raise CodexCredentialsError(
                f"Refresh token is no longer valid ({_error_code(error_body)}). "
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
        return json.loads(error_body).get("error")
    except (json.JSONDecodeError, AttributeError):
        return None


if __name__ == "__main__":
    print(f"Codex home: {codex_home()}")
    print(f"Auth file:  {codex_auth_path()} (exists: {codex_auth_path().exists()})")

    credentials = load_codex_credentials()
    token = credentials.access_token
    expires_in = credentials.expires_in

    print(f"Loaded from: {credentials.auth_path}")
    print(f"Account ID:  {credentials.account_id or '-'}")
    print(f"Token:       {token[:12]}...{token[-6:]} (len {len(token)})")
    print(f"Refreshed:   {credentials.refreshed}")
    if expires_in is not None:
        print(f"Expires in:  {expires_in / 60:.1f} min")
