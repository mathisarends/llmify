"""Codex provider and Codex CLI authentication support."""

_AUTH_EXPORTS = {
    "CodexCliAuth",
    "CodexCredentials",
    "CodexCredentialsError",
    "codex_auth_path",
    "codex_home",
    "read_codex_credentials",
    "refresh_codex_credentials",
}
_CLIENT_EXPORTS = {"ChatCodex"}


def __getattr__(name: str):
    if name in _AUTH_EXPORTS:
        from . import auth

        return getattr(auth, name)
    if name in _CLIENT_EXPORTS:
        from . import client

        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})


__all__ = [
    "ChatCodex",
    "CodexCliAuth",
    "CodexCredentials",
    "CodexCredentialsError",
    "codex_auth_path",
    "codex_home",
    "read_codex_credentials",
    "refresh_codex_credentials",
]
