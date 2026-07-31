"""Guardrail tests for fenced StandardNameSource-to-name linking.

Compatibility helpers without ownership fences must not mutate the graph.
Claim-aware graph helpers must match the exact source fence and target name
before changing source state, mirror ``produced_sn_id``, and create the
``PRODUCED_NAME`` relationship.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.graph_ops import (
    mark_sources_attached,
    mark_sources_composed,
)
from imas_codex.standard_names.workers import (
    _update_sources_after_attach,
    _update_sources_after_compose,
)


def _captured_query(mock_gc: MagicMock) -> str:
    """Return the Cypher string passed to ``gc.query`` on the mock."""
    assert mock_gc.return_value.__enter__.return_value.query.called, (
        "GraphClient.query was not invoked"
    )
    call = mock_gc.return_value.__enter__.return_value.query.call_args
    # The positional arg is the query string; support keyword form too.
    if call.args:
        return call.args[0]
    return call.kwargs["query"]


def _assert_match_before_set(cypher: str) -> None:
    """The sn MATCH must appear before the sns SET so status gates on sn."""
    stripped = " ".join(cypher.split())
    sn_match_idx = stripped.find("MATCH (sn:StandardName")
    sns_set_idx = stripped.find("SET sns.status")
    assert sn_match_idx != -1, f"missing MATCH (sn:StandardName in Cypher:\n{cypher}"
    assert sns_set_idx != -1, f"missing SET sns.status in Cypher:\n{cypher}"
    assert sn_match_idx < sns_set_idx, (
        "SET sns.status must not run before MATCH (sn:StandardName) — "
        "otherwise rows with no matching SN still bump status and leave "
        f"orphan sources. Cypher:\n{cypher}"
    )


def _assert_writes_produced_sn_id(cypher: str) -> None:
    assert "produced_sn_id" in cypher, (
        f"Cypher must set sns.produced_sn_id for recoverability:\n{cypher}"
    )


def _assert_merges_produced_edge(cypher: str) -> None:
    assert "PRODUCED_NAME" in cypher, (
        f"Cypher must MERGE the (:StandardNameSource)-[:PRODUCED_NAME]->(:StandardName) "
        f"edge:\n{cypher}"
    )


class TestWorkerHelpers:
    """Unfenced compatibility helpers reject source-id-only mutations."""

    def _fake_candidates(self) -> list[dict]:
        return [
            {
                "source_id": "equilibrium/time_slice/profiles_1d/psi",
                "id": "poloidal_magnetic_flux",
            },
            {
                "source_id": "core_profiles/profiles_1d/electrons/temperature",
                "id": "electron_temperature",
            },
        ]

    def test_compose_without_fence_does_not_open_graph(self) -> None:
        log = logging.getLogger("test")
        adapter = logging.LoggerAdapter(log, {})
        with patch("imas_codex.graph.client.GraphClient") as MockGC:  # noqa: N806
            _update_sources_after_compose(self._fake_candidates(), "dd", adapter)

        assert not MockGC.called

    def test_attach_without_fence_does_not_open_graph(self) -> None:
        log = logging.getLogger("test")
        adapter = logging.LoggerAdapter(log, {})

        class A:
            def __init__(self, sid: str, name: str) -> None:
                self.source_id = sid
                self.standard_name = name

        attachments = [
            A("equilibrium/time_slice/global_quantities/ip", "plasma_current")
        ]
        with patch("imas_codex.graph.client.GraphClient") as MockGC:  # noqa: N806
            _update_sources_after_attach(attachments, "dd", adapter)

        assert not MockGC.called

    def test_compose_warns_when_fence_is_missing(self, caplog) -> None:
        log = logging.getLogger("test_linking")
        adapter = logging.LoggerAdapter(log, {})
        with patch("imas_codex.graph.client.GraphClient") as MockGC:  # noqa: N806
            with caplog.at_level(logging.WARNING, logger="test_linking"):
                _update_sources_after_compose(self._fake_candidates(), "dd", adapter)

        assert any("unfenced" in r.message for r in caplog.records)
        assert not MockGC.called


class TestGraphOpsHelpers:
    """Cover the token-verified ``mark_sources_(composed|attached)`` helpers."""

    def test_mark_sources_composed_cypher_invariants(self) -> None:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
            MockGC.return_value.__enter__.return_value.query.return_value = [
                {"affected": 2}
            ]
            affected = mark_sources_composed(
                token="tok-123",
                claim_seq=3,
                source_ids=["dd:a", "dd:b"],
                standard_name_id="electron_temperature",
            )
        assert affected == 2
        cypher = _captured_query(MockGC)
        _assert_match_before_set(cypher)
        _assert_writes_produced_sn_id(cypher)
        _assert_merges_produced_edge(cypher)
        assert "sns.claim_seq = $claim_seq" in cypher

    def test_mark_sources_attached_cypher_invariants(self) -> None:
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:  # noqa: N806
            MockGC.return_value.__enter__.return_value.query.return_value = [
                {"affected": 1}
            ]
            affected = mark_sources_attached(
                token="tok-abc",
                claim_seq=4,
                source_ids=["dd:x"],
                standard_name_id="plasma_current",
            )
        assert affected == 1
        cypher = _captured_query(MockGC)
        _assert_match_before_set(cypher)
        _assert_writes_produced_sn_id(cypher)
        _assert_merges_produced_edge(cypher)
        assert "sns.claim_seq = $claim_seq" in cypher


# =====================================================================
# Error-sibling skip: deterministic siblings have no StandardNameSource
# =====================================================================


def test_update_sources_rejects_unfenced_error_siblings():
    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    candidates = [
        # Regular candidate — should be linked
        {
            "id": "elongation",
            "source_id": "equilibrium/time_slice/profiles_1d/elongation",
            "model": "openrouter/anthropic/claude-sonnet-4.6",
        },
        # Error sibling — must be skipped
        {
            "id": "upper_uncertainty_of_elongation",
            "source_id": "equilibrium/time_slice/profiles_1d/elongation_error_upper",
            "model": "deterministic:dd_error_modifier",
        },
        {
            "id": "lower_uncertainty_of_elongation",
            "source_id": "equilibrium/time_slice/profiles_1d/elongation_error_lower",
            "model": "deterministic:dd_error_modifier",
        },
    ]
    with patch("imas_codex.graph.client.GraphClient") as mock_gc:
        _update_sources_after_compose(candidates, "dd", wlog)

    assert not mock_gc.called


def test_update_sources_all_error_siblings_no_query():
    """When every candidate is an error sibling, no query should be issued."""
    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    candidates = [
        {
            "id": f"upper_uncertainty_of_{base}",
            "source_id": f"equilibrium/time_slice/profiles_1d/{base}_error_upper",
            "model": "deterministic:dd_error_modifier",
        }
        for base in ("elongation", "triangularity")
    ]
    with patch("imas_codex.graph.client.GraphClient") as mock_gc:
        _update_sources_after_compose(candidates, "dd", wlog)
    assert not mock_gc.return_value.__enter__.return_value.query.called, (
        "No source linking query should run when batch is empty after filtering"
    )
