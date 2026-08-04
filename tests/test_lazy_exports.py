"""Guards the lazy re-export layer of ``llmify`` and ``llmify.providers``.

Both packages resolve their public API through ``__getattr__`` so that importing
``llmify`` does not pull in every optional provider SDK. The downside is that a
name added to ``__all__`` without the matching ``__getattr__`` branch stays
invisible until a user hits the ``AttributeError`` at runtime.
"""

import pytest

import llmify
from llmify import providers


def _resolve(module: object, name: str) -> object:
    try:
        return getattr(module, name)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {name!r}: {exc}")


class TestPublicApiResolves:
    @pytest.mark.parametrize("name", sorted(llmify.__all__))
    def test_top_level_name_resolves(self, name: str) -> None:
        assert _resolve(llmify, name) is not None

    @pytest.mark.parametrize("name", sorted(providers.__all__))
    def test_providers_name_resolves(self, name: str) -> None:
        assert _resolve(providers, name) is not None

    def test_top_level_all_has_no_duplicates(self) -> None:
        assert sorted(llmify.__all__) == sorted(set(llmify.__all__))

    def test_providers_all_has_no_duplicates(self) -> None:
        assert sorted(providers.__all__) == sorted(set(providers.__all__))

    def test_providers_are_a_subset_of_the_top_level_api(self) -> None:
        assert set(providers.__all__) <= set(llmify.__all__)


class TestUnknownAttributes:
    def test_top_level_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'NotARealExport'"):
            getattr(llmify, "NotARealExport")

    def test_providers_raises_attribute_error(self) -> None:
        with pytest.raises(AttributeError, match="has no attribute 'NotARealExport'"):
            getattr(providers, "NotARealExport")

    @pytest.mark.parametrize("module", [llmify, providers])
    def test_dir_exposes_the_lazy_exports(self, module: object) -> None:
        # PEP 562: __getattr__ alone leaves the lazy names out of dir(), which
        # breaks REPL and IDE completion.
        assert set(module.__all__) <= set(dir(module))
