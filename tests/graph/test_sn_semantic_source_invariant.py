"""Semantic-source ledger invariant for StandardNameSource provenance.

Runs against a live Neo4j graph alongside the other ``tests/graph/`` tests:

    uv run pytest tests/graph/test_sn_semantic_source_invariant.py -m graph -rA

A ``StandardNameSource`` that has been composed or attached records *which*
standard name its DD path / facility signal currently supports. That record is
kept in three places which must agree:

1. the ``PRODUCED_NAME`` edge — the source of truth, and single-valued: exactly
   one *live* target (a target at ``superseded``/``exhausted`` is dead history);
2. the ``produced_sn_id`` scalar mirror, denormalised for cheap filtering;
3. the upstream projection — the backing ``IMASNode``/``FacilitySignal`` reached
   via ``FROM_DD_PATH``/``FROM_SIGNAL`` must carry a ``HAS_STANDARD_NAME`` edge
   to that same name.

Every name-changing route (refine, edit, compact, cascade rename) is required to
move all three together — that is what ``retarget_standard_name_sources`` and
``bind_sources_exclusively`` exist to guarantee. A disagreement means a writer
moved one mirror and stranded another, so the ledger no longer answers "what
supports this name?" consistently.

The check itself lives with the operations it guards
(:func:`imas_codex.standard_names.provenance_lifecycle.find_semantic_source_invariant_violations`);
this module is what gives it signal.

If the graph has fewer than 10 accepted StandardName nodes the module is skipped
(mirrors ``test_sn_graph.py`` / ``test_sn_edge_integrity.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.provenance_lifecycle import (
    find_semantic_source_invariant_violations,
)

pytestmark = pytest.mark.graph


@pytest.fixture(scope="module")
def sn_graph(graph_client):
    """Shared session GraphClient; skip if the SN corpus is too thin to judge."""
    rows = graph_client.query(
        "MATCH (sn:StandardName {name_stage: 'accepted'}) RETURN count(sn) AS n"
    )
    accepted = rows[0]["n"] if rows else 0
    if accepted < 10:
        pytest.skip(
            f"Graph has only {accepted} accepted StandardName nodes (<10); "
            "populate via `sn run` before running SN provenance-ledger tests."
        )
    return graph_client


def _classify(violation: dict[str, Any]) -> str:
    """Name the disagreeing mirror so a failure points at the guilty writer."""
    live = violation.get("live_targets") or []
    if not live:
        return "no live PRODUCED_NAME target (every target is dead history)"
    if len(live) > 1:
        return "more than one live PRODUCED_NAME target"
    if violation.get("produced_sn_id") != live[0]:
        return "produced_sn_id scalar disagrees with the PRODUCED_NAME edge"
    if live[0] not in (violation.get("mapped_ids") or []):
        return "backing DD path / signal lacks the HAS_STANDARD_NAME projection"
    return "unclassified mirror disagreement"


def _describe(violation: dict[str, Any]) -> str:
    return (
        f"{violation.get('source_id')}: {_classify(violation)} "
        f"[live={violation.get('live_targets')}, "
        f"scalar={violation.get('produced_sn_id')!r}, "
        f"projected={violation.get('mapped_ids')}]"
    )


class TestSemanticSourceLedgerInvariant:
    """The three current-target mirrors of a live semantic source must agree."""

    def test_no_semantic_source_mirror_disagreements(self, sn_graph):
        """Every composed/attached source has exactly one agreed current target.

        Scoped to ``status IN ['composed', 'attached']`` — a source still at
        ``extracted``, or retired to ``vocab_gap``/``failed``/``stale``, has no
        current target to agree about. For the in-scope sources the assertion is
        the full invariant: one live ``PRODUCED_NAME`` target, the
        ``produced_sn_id`` scalar equal to it, and the backing DD path / signal
        projecting ``HAS_STANDARD_NAME`` onto it.

        The failure message groups offenders by which mirror disagrees, so the
        count per class points straight at the writer that stranded it.
        """
        violations = find_semantic_source_invariant_violations(sn_graph)
        if not violations:
            return

        by_class: dict[str, int] = {}
        for violation in violations:
            label = _classify(violation)
            by_class[label] = by_class.get(label, 0) + 1
        summary = "; ".join(
            f"{count}x {label}"
            for label, count in sorted(by_class.items(), key=lambda kv: -kv[1])
        )
        raise AssertionError(
            f"{len(violations)} composed/attached StandardNameSource(s) have "
            f"disagreeing current-target mirrors ({summary}). First offenders:\n  "
            + "\n  ".join(_describe(v) for v in violations[:25])
        )
