"""Tests for the auto grammar-sync helper (`sn run` startup) and sync idempotency.

Grammar sync is no longer a standalone CLI command — it is auto-run at
``sn run`` startup (via ``_auto_sync_grammar``) and re-seeded by
``sn clear``. These tests cover the auto-sync helper's version-gating and
graceful-degradation behaviour, plus the ISN ``sync_grammar`` idempotency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

from imas_codex.cli.sn import _auto_sync_grammar


@dataclass
class FakeReport:
    """Fake SyncReport matching ISN's dataclass shape."""

    version: str = "0.7.0rc10"
    applied: bool = False
    created_version: bool = True
    segments_written: int = 11
    tokens_written: int = 7
    templates_written: int = 6
    next_edges_written: int = 10
    defines_edges_written: int = 4
    has_token_edges_written: int = 3
    elapsed_seconds: float = 0.01
    planned_statements: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _fake_report():
    """Build a fake SyncReport-like object."""
    return FakeReport()


def _fake_spec() -> dict[str, Any]:
    return {
        "version": "0.7.0rc10",
        "segments": [{"name": "subject"}],
        "templates": [],
    }


def _mock_gc(active_version: str | None) -> MagicMock:
    """A context-manager GraphClient whose active-version query returns ``active_version``."""
    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = None
    gc.query = MagicMock(
        return_value=[{"version": active_version}] if active_version else []
    )
    return gc


def test_auto_sync_skips_when_in_sync():
    """When the graph's active grammar matches the installed ISN, sync is a no-op."""
    from imas_standard_names import __version__ as isn_version

    gc = _mock_gc(isn_version)
    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
        ) as mock_sync,
    ):
        _auto_sync_grammar(quiet=True)

    mock_sync.assert_not_called()


def test_auto_sync_runs_when_version_differs():
    """A version mismatch triggers sync_isn_grammar_to_graph with the open client."""
    gc = _mock_gc("0.0.1-stale")
    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.grammar_sync.sync_isn_grammar_to_graph"
        ) as mock_sync,
    ):
        _auto_sync_grammar(quiet=True)

    mock_sync.assert_called_once()
    _, kwargs = mock_sync.call_args
    assert kwargs.get("gc") is gc  # reuses the open client


def test_auto_sync_degrades_gracefully_on_failure():
    """A graph/sync failure is swallowed (logged) — the run is never crashed."""

    def _boom(*_a, **_kw):
        raise ConnectionError("Neo4j unreachable")

    with patch("imas_codex.graph.client.GraphClient", side_effect=_boom):
        # Must NOT raise.
        _auto_sync_grammar(quiet=True)


# ---------------------------------------------------------------------------
# Sync idempotency tests (plan 29 E.8)
# ---------------------------------------------------------------------------


def _make_mock_spec():
    """Return a fixed grammar graph spec for idempotency testing."""
    return {
        "version": "0.7.0rc10",
        "segment_order": ["subject", "physical_base", "object"],
        "segments": [
            {
                "name": "subject",
                "position": 0,
                "required": False,
                "tokens": [
                    {"value": "electron", "aliases": []},
                    {"value": "ion", "aliases": []},
                ],
            },
            {
                "name": "physical_base",
                "position": 1,
                "required": True,
                "tokens": [],  # open vocab
            },
            {
                "name": "object",
                "position": 2,
                "required": False,
                "tokens": [
                    {"value": "plasma", "aliases": []},
                ],
            },
        ],
        "templates": [
            {"name": "of_object", "segment": "object", "pattern": "of_{token}"},
        ],
    }


def _run_sync_with_mock(mock_gc: MagicMock, spec: dict) -> FakeReport:
    """Invoke sync_grammar with mocked spec and GraphClient, return report."""
    from imas_standard_names.graph.sync import sync_grammar

    with patch(
        "imas_standard_names.graph.sync.get_grammar_graph_spec", return_value=spec
    ):
        return sync_grammar(mock_gc, active_version="0.7.0rc10")


class TestSyncIdempotency:
    """Verify that running sync twice produces no duplicates (E.8)."""

    def test_second_sync_no_new_statements(self) -> None:
        """Running sync twice should produce the same Cypher MERGE statements."""
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _run_sync_with_mock(mock_gc, spec)
        calls_after_first = len(mock_gc.query.call_args_list)

        _run_sync_with_mock(mock_gc, spec)
        calls_after_second = len(mock_gc.query.call_args_list)

        first_count = calls_after_first
        second_count = calls_after_second - calls_after_first
        assert first_count == second_count, (
            f"First run issued {first_count} statements, "
            f"second run issued {second_count}. "
            f"Expected identical MERGE-based statements."
        )

    def test_second_sync_same_version(self) -> None:
        """Both sync runs should report the same version."""
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        report1 = _run_sync_with_mock(mock_gc, spec)
        report2 = _run_sync_with_mock(mock_gc, spec)

        assert report1.version == report2.version == "0.7.0rc10"

    def test_merge_cypher_used_not_create(self) -> None:
        """All node/edge writes must use MERGE, never CREATE."""
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _run_sync_with_mock(mock_gc, spec)

        for call_obj in mock_gc.query.call_args_list:
            cypher = call_obj[0][0] if call_obj[0] else ""
            if "CONSTRAINT" in cypher or "INDEX" in cypher:
                continue
            if "GrammarSegment" in cypher or "GrammarToken" in cypher:
                assert "MERGE" in cypher, (
                    f"Expected MERGE in Cypher, got CREATE-style: {cypher[:120]}"
                )

    def test_version_node_count_stable(self) -> None:
        """After two syncs the ISNGrammarVersion MERGE should target the same id."""
        spec = _make_mock_spec()
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])

        _run_sync_with_mock(mock_gc, spec)
        _run_sync_with_mock(mock_gc, spec)

        version_merges = [
            c
            for c in mock_gc.query.call_args_list
            if "MERGE" in str(c[0][0])
            and "ISNGrammarVersion" in str(c[0][0])
            and "MERGE (v:ISNGrammarVersion" in str(c[0][0])
        ]
        assert len(version_merges) == 2, (
            f"Expected 2 ISNGrammarVersion MERGEs (one per sync), "
            f"got {len(version_merges)}"
        )
        versions = {c[1].get("version", "") for c in version_merges}
        assert len(versions) == 1, (
            f"Expected same version for both syncs, got {versions}"
        )
