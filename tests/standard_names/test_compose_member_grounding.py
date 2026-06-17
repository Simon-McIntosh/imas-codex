"""Per-member DD-documentation grounding in the compose enrichment step.

A compose candidate's name can span N sibling DD leaves (``family_siblings``,
cross-IDS equivalents). ``_enrich_batch_items`` grounds the primary ``path`` on
its rich enriched description; this module verifies the SEPARATE per-member
``dd_paths_docs`` {path: terse_doc} map it builds so the name is grounded on
every leaf it covers — terse DD doc only, bounded, primary excluded, empty
docs skipped.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch

from imas_codex.standard_names import workers
from imas_codex.standard_names.workers import (
    _MAX_MEMBER_DOCS,
    _enrich_batch_items,
)


class _FakeGraph:
    """Minimal GraphClient stand-in.

    The per-item DD-context query (``path=`` kwarg) returns nothing so the
    per-item enrichment short-circuits, leaving the pre-set member fields
    intact; the member-doc query (``paths=`` kwarg) returns the supplied docs.
    """

    def __init__(self, member_docs: list[dict]) -> None:
        self._member_docs = member_docs

    def query(self, _cypher: str, **kwargs):
        if "paths" in kwargs:
            return list(self._member_docs)
        return []


@contextmanager
def _patched_graph(member_docs: list[dict]):
    fake = _FakeGraph(member_docs)

    @contextmanager
    def _cm():
        yield fake

    # GraphClient is imported locally inside _enrich_batch_items. The hybrid
    # batch returns one result per tuple (strict zip) — keep that contract.
    with (
        patch("imas_codex.graph.client.GraphClient", _cm),
        patch.object(
            workers,
            "_hybrid_search_neighbours_batch",
            side_effect=lambda _gc, tuples: [None] * len(tuples),
        ),
    ):
        yield


def test_member_docs_attached_and_empty_skipped() -> None:
    items = [
        {
            "path": "magnetics/ip",
            "family_siblings": ["magnetics/rogowski_coil/current"],
            "cross_ids_paths": ["equilibrium/time_slice/global_quantities/ip"],
        }
    ]
    member_docs = [
        {
            "path": "magnetics/rogowski_coil/current",
            "documentation": "Rogowski current",
        },
        # Empty doc must be skipped, not rendered as a blank clause.
        {"path": "equilibrium/time_slice/global_quantities/ip", "documentation": ""},
    ]
    with _patched_graph(member_docs):
        _enrich_batch_items(items)

    docs = items[0]["dd_paths_docs"]
    assert docs == {"magnetics/rogowski_coil/current": "Rogowski current"}


def test_primary_path_excluded_from_members() -> None:
    items = [
        {
            "path": "a/b",
            # Self-reference must never appear as its own member.
            "family_siblings": ["a/b", "a/c"],
        }
    ]
    member_docs = [
        {"path": "a/b", "documentation": "primary doc"},
        {"path": "a/c", "documentation": "sibling doc"},
    ]
    with _patched_graph(member_docs):
        _enrich_batch_items(items)

    docs = items[0]["dd_paths_docs"]
    assert "a/b" not in docs
    assert docs == {"a/c": "sibling doc"}


def test_member_count_is_bounded() -> None:
    siblings = [f"ids/leaf_{i}" for i in range(_MAX_MEMBER_DOCS + 5)]
    items = [{"path": "ids/primary", "family_siblings": siblings}]
    member_docs = [{"path": p, "documentation": f"doc {p}"} for p in siblings]
    with _patched_graph(member_docs):
        _enrich_batch_items(items)

    docs = items[0]["dd_paths_docs"]
    assert len(docs) == _MAX_MEMBER_DOCS


def test_no_members_no_key() -> None:
    items = [{"path": "lonely/leaf"}]
    with _patched_graph([]):
        _enrich_batch_items(items)
    assert "dd_paths_docs" not in items[0]


def test_long_member_doc_truncated() -> None:
    long_doc = "x" * 1000
    items = [{"path": "p", "family_siblings": ["q"]}]
    member_docs = [{"path": "q", "documentation": long_doc}]
    with _patched_graph(member_docs):
        _enrich_batch_items(items)

    rendered = items[0]["dd_paths_docs"]["q"]
    assert len(rendered) < len(long_doc)
    assert rendered.endswith("…")
