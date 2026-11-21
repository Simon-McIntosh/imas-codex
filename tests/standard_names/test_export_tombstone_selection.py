"""Retired (superseded) identities are never selected as export candidates.

A tombstoned identity must stop at the population boundary: it is not part of
the candidate set at all, so the export never drops it and the accounting
never counts it as an exclusion. Both selection queries — the raw candidate
fetch and the accounting population fetch — must carry the same exclusion so
eligibility and the candidate count agree.
"""

from __future__ import annotations

from unittest.mock import patch

from iter_codex.standard_names.export import (
    _fetch_candidates,
    _fetch_export_population,
    _tombstone_exclusion_clause,
)


class _CapturingGraphClient:
    """GraphClient stand-in that records the last emitted Cypher text."""

    def __init__(self):
        self.cypher = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def query(self, cypher: str, **params):
        self.cypher = cypher
        return []


def _capture(query_builder):
    """Run one selection query against a fake graph and return its Cypher."""
    client = _CapturingGraphClient()
    cls = _CapturingGraphClient

    class _EnteringGraphClient(cls):
        def __enter__(self):
            return client

    with patch(
        "iter_codex.graph.client.GraphClient",
        _EnteringGraphClient,
    ):
        query_builder()
    return client.cypher


def test_tombstone_clause_suppresses_both_retirement_fields() -> None:
    clause = _tombstone_exclusion_clause()

    assert "coalesce(sn.status, '') <> 'superseded'" in clause
    assert "coalesce(sn.name_stage, '') <> 'superseded'" in clause


def test_candidate_fetch_excludes_tombstones_in_batch_scope() -> None:
    cypher = _capture(
        lambda: _fetch_candidates(
            batch=["normalized_toroidal_beta", "area_of_diagnostic_aperture"]
        )
    )

    assert _tombstone_exclusion_clause() in cypher
    assert "sn.id IN $batch" in cypher


def test_candidate_fetch_excludes_tombstones_in_full_scope() -> None:
    cypher = _capture(lambda: _fetch_candidates())

    assert _tombstone_exclusion_clause() in cypher
    assert "name_stage IN ['accepted', 'approved']" in cypher


def test_population_fetch_excludes_tombstones_in_batch_scope() -> None:
    cypher = _capture(
        lambda: _fetch_export_population(
            batch=["normalized_toroidal_beta"],
            require_docs_review=False,
        )
    )

    assert _tombstone_exclusion_clause() in cypher
    assert "sn.id IN $batch" in cypher
