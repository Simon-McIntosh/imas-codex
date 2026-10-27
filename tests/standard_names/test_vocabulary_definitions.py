"""The vocabulary definition map and its two consumers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from imas_codex.standard_names import segment_selection, vocabulary_definitions
from imas_codex.standard_names.worker_tasks import (
    worker_review_names as _owning_worker_review_names,
)


@pytest.fixture(autouse=True)
def _fresh_definitions():
    loader = vocabulary_definitions.vocabulary_definitions
    loader.cache_clear()
    yield
    loader.cache_clear()


def _entry(definition: str) -> SimpleNamespace:
    return SimpleNamespace(definition=definition)


def test_undefined_tokens_are_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    from imas_standard_names.grammar import vocab_loaders

    monkeypatch.setattr(
        vocab_loaders,
        "load_physical_bases",
        lambda: SimpleNamespace(
            bases={"defined": _entry("A meaning"), "empty": _entry("")}
        ),
    )
    monkeypatch.setattr(
        vocab_loaders,
        "load_geometry_carriers",
        lambda: SimpleNamespace(carriers={"blank": _entry("   ")}),
    )

    assert vocabulary_definitions.vocabulary_definitions() == {"defined": "A meaning"}


def test_loci_are_included_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    from imas_standard_names.grammar import vocab_loaders

    monkeypatch.setattr(
        vocab_loaders,
        "load_physical_bases",
        lambda: SimpleNamespace(bases={"base": _entry("Base meaning")}),
    )
    monkeypatch.setattr(
        vocab_loaders,
        "load_geometry_carriers",
        lambda: SimpleNamespace(carriers={}),
    )
    monkeypatch.setattr(
        vocab_loaders,
        "load_locus_registry",
        lambda: SimpleNamespace(loci={"place": _entry("Place meaning")}),
    )

    assert vocabulary_definitions.vocabulary_definitions() == {"base": "Base meaning"}
    assert vocabulary_definitions.vocabulary_definitions(include_loci=True) == {
        "base": "Base meaning",
        "place": "Place meaning",
    }


def test_selection_uses_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def sentinel(*, include_loci: bool = False) -> dict[str, str]:
        calls.append(include_loci)
        return {"beta": "Sentinel meaning"}

    monkeypatch.setattr(vocabulary_definitions, "vocabulary_definitions", sentinel)

    assert segment_selection.option_text("beta") == "beta — Sentinel meaning"
    assert calls == [True]


def test_review_gate_uses_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []

    def sentinel(*, include_loci: bool = False) -> dict[str, str]:
        calls.append(include_loci)
        return {"beta": "Sentinel meaning"}

    monkeypatch.setattr(vocabulary_definitions, "vocabulary_definitions", sentinel)

    assert (
        _owning_worker_review_names.semantic_gate_name_text("beta")
        == "Sentinel meaning"
    )
    assert calls == [False]
