"""Grammar snapshot selection follows package-version semantics."""

from unittest.mock import MagicMock

from imas_standard_names import get_grammar_context

from imas_codex.standard_names.grammar_query import (
    order_grammar_versions,
    select_grammar_version,
)
from imas_codex.standard_names.grammar_sync import _grammar_context_token_rows
from imas_codex.standard_names.graph_ops import _resolve_grammar_token_version


def test_release_candidate_versions_are_ordered_numerically() -> None:
    """A one-digit candidate never outranks later two-digit candidates."""
    versions = ["0.8.0rc9", "0.8.0rc65", "0.8.0rc66"]

    assert order_grammar_versions(versions) == [
        "0.8.0rc66",
        "0.8.0rc65",
        "0.8.0rc9",
    ]
    assert (
        select_grammar_version(
            {"version": version, "active": False} for version in versions
        )
        == "0.8.0rc66"
    )


def test_active_snapshot_precedes_semantically_newer_fallback() -> None:
    rows = [
        {"version": "0.8.0rc65", "active": True},
        {"version": "0.8.0rc66", "active": False},
    ]

    assert select_grammar_version(rows) == "0.8.0rc65"


def test_graph_resolver_uses_active_snapshot_when_runtime_is_absent() -> None:
    gc = MagicMock()
    gc.query.side_effect = [
        [],
        [
            {"version": "0.8.0rc9", "active": False},
            {"version": "0.8.0rc65", "active": True},
            {"version": "0.8.0rc66", "active": False},
        ],
    ]

    assert _resolve_grammar_token_version(gc, "0.8.0rc67") == "0.8.0rc65"


def test_exact_runtime_snapshot_precedes_active_fallback() -> None:
    rows = [
        {"version": "0.8.0rc65", "active": True},
        {"version": "0.8.0rc66", "active": False},
    ]

    assert select_grammar_version(rows, preferred_version="0.8.0rc66") == ("0.8.0rc66")


def test_public_context_tokens_cover_every_declared_segment() -> None:
    context = get_grammar_context()
    rows = _grammar_context_token_rows(context)

    assert {row["segment"] for row in rows} == set(context["segment_descriptions"])
    assert sum(row["segment"] == "physical_base" for row in rows) == len(
        context["grammar"]["vocabularies"]["physical_bases"]
    )
