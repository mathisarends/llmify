"""Guards the public exports of providers implemented as packages."""

from types import ModuleType

import pytest

from llmify.providers import anthropic, codex, google, openai_responses

PROVIDER_PACKAGES = (anthropic, codex, google, openai_responses)


def _resolve(module: ModuleType, name: str) -> object:
    try:
        return getattr(module, name)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {module.__name__}.{name}: {exc}")


@pytest.mark.parametrize(
    ("provider", "name"),
    [
        (provider, name)
        for provider in PROVIDER_PACKAGES
        for name in sorted(provider.__all__)
    ],
)
def test_public_provider_export_resolves(provider: ModuleType, name: str) -> None:
    assert _resolve(provider, name) is not None


@pytest.mark.parametrize("provider", PROVIDER_PACKAGES)
def test_public_provider_exports_are_unique_and_visible(provider: ModuleType) -> None:
    assert len(provider.__all__) == len(set(provider.__all__))
    assert set(provider.__all__) <= set(dir(provider))


def test_openai_responses_exposes_short_submodule_paths() -> None:
    assert openai_responses.transport.__name__.endswith(".transport")
    assert openai_responses.types.__name__.endswith(".types")
