"""Tests for the shared documentation name-reference parser."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.doc_links import find_name_references


def test_math_notation_is_not_a_name_reference() -> None:
    documentation = r"The coefficient is $$C[f_q]$$ and also \[C[f_q]\]."

    assert find_name_references(documentation) == ()


def test_bare_and_inline_name_references_are_detected() -> None:
    references = find_name_references(
        "Compare [poloidal_magnetic_flux] with name:toroidal_magnetic_flux.",
        known={"poloidal_magnetic_flux"},
    )

    assert [(reference.name, reference.syntax) for reference in references] == [
        ("poloidal_magnetic_flux", "bare_bracket"),
        ("toroidal_magnetic_flux", "inline_name"),
    ]


def test_known_none_returns_candidates_without_resolving() -> None:
    references = find_name_references("See [not_yet_resolved].", known=None)

    assert [(reference.name, reference.syntax) for reference in references] == [
        ("not_yet_resolved", "bare_bracket")
    ]


def test_resolver_failure_propagates() -> None:
    def unavailable(_candidates: tuple[str, ...]) -> set[str]:
        raise RuntimeError("catalog unavailable")

    with pytest.raises(RuntimeError, match="catalog unavailable"):
        find_name_references("See [poloidal_magnetic_flux].", known=unavailable)


def test_non_name_markdown_constructs_are_not_bare_references() -> None:
    references = find_name_references(
        "![plot_label](figure.png) and "
        "[readable label](https://example.test) and "
        "[flux](name:poloidal_magnetic_flux) and `values[index_name]`"
    )

    assert [(reference.name, reference.syntax) for reference in references] == [
        ("poloidal_magnetic_flux", "inline_name")
    ]
