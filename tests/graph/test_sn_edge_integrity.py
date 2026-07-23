"""StandardName graph edge-integrity invariants.

Runs against a live Neo4j graph alongside other ``tests/graph/`` tests:

    uv run pytest tests/graph/test_sn_edge_integrity.py -m graph -rA

These assert structural invariants on the StandardName↔DD/Unit edge fabric —
distinct from the metric gates in ``test_sn_graph.py`` (corpus quality) and the
unit-value cross-checks in ``test_sn_unit_integrity.py`` (``sn.unit`` vs its
HAS_UNIT target). The invariants here guard against edge-writer defects that
let an SN accrete duplicate/stale HAS_UNIT edges, lose its unit edge, carry a
stale denormalised ``source_paths`` scalar, or attach to a DD path whose unit
dimensionally disagrees with the name.

If the graph has fewer than 10 accepted StandardName nodes the whole module is
skipped (mirrors ``test_sn_graph.py``).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.graph

# Terminal / dead name stages — refined-away or given-up names that are no
# longer canonical and are legitimately allowed to trail incomplete edges.
_TERMINAL_STAGES = ["superseded", "exhausted", "contested"]


def _connect_and_count_accepted():
    """Return (GraphClient, accepted_count) or raise on connection failure."""
    from imas_codex.graph.client import GraphClient

    gc = GraphClient()
    rows = gc.query(
        "MATCH (sn:StandardName {name_stage: 'accepted'}) RETURN count(sn) AS n"
    )
    return gc, rows[0]["n"]


@pytest.fixture(scope="module")
def gc():
    """Module-scoped GraphClient; skip if too few accepted names."""
    client, count = _connect_and_count_accepted()
    if count < 10:
        pytest.skip(
            f"Graph has only {count} accepted StandardName nodes (<10); "
            "populate via `sn run` before running SN edge-integrity tests."
        )
    yield client


class TestStandardNameEdgeIntegrity:
    """Structural invariants on the StandardName edge fabric."""

    def test_at_most_one_has_unit_edge_per_name(self, gc):
        """No live StandardName carries more than one HAS_UNIT edge.

        An SN's dimensionality is single-valued; the writer self-heals
        (drop-existing-then-MERGE) so at most one HAS_UNIT edge should exist.
        Superseded/exhausted/contested names are excluded — they are dead and
        may trail stale edges. A failure here means a bare ``MERGE`` writer
        left a second edge to a different Unit.
        """
        rows = gc.query(
            """
            MATCH (sn:StandardName)-[r:HAS_UNIT]->(u:Unit)
            WHERE NOT (sn.name_stage IN $terminal)
            WITH sn, count(r) AS edges, collect(u.id) AS units
            WHERE edges > 1
            RETURN sn.id AS name, edges, units
            ORDER BY name
            LIMIT 25
            """,
            terminal=_TERMINAL_STAGES,
        )
        assert not rows, (
            f"{len(rows)} live StandardName(s) have >1 HAS_UNIT edge: "
            + "; ".join(f"{r['name']} → {r['units']}" for r in rows)
        )

    def test_live_name_with_unit_property_has_unit_edge(self, gc):
        """Every non-terminal SN with a ``unit`` property has a HAS_UNIT edge.

        A validated live name whose ``unit`` scalar is set but which carries
        zero HAS_UNIT edges is an edge-creation gap (or a refine that stranded
        the edge). Dimensionless names correctly store ``unit='1'`` with an
        edge to the ``'1'`` Unit, so this covers them too. Terminal stages are
        excluded.
        """
        rows = gc.query(
            """
            MATCH (sn:StandardName)
            WHERE sn.unit IS NOT NULL
              AND NOT (sn.name_stage IN $terminal)
              AND NOT (sn)-[:HAS_UNIT]->(:Unit)
            RETURN sn.id AS name, sn.unit AS unit, sn.name_stage AS stage
            ORDER BY name
            LIMIT 25
            """,
            terminal=_TERMINAL_STAGES,
        )
        assert not rows, (
            f"{len(rows)} live StandardName(s) have a unit property but no "
            "HAS_UNIT edge: "
            + "; ".join(f"{r['name']} (unit={r['unit']}, {r['stage']})" for r in rows)
        )

    def test_source_paths_scalar_consistent_with_edges(self, gc):
        """``sn.source_paths`` contains no entry absent from the live edges.

        The denormalised scalar must be a subset of the DD paths reachable via
        the live provenance edges — ``HAS_STANDARD_NAME`` from an IMASNode and
        ``PRODUCED_NAME`` from a DD-typed StandardNameSource. Entries that
        appear only in the scalar are stale residue from pruned/refined
        mappings; the edges are the source of truth. Scoped to accepted names
        (the ones a consumer trusts).
        """
        rows = gc.query(
            """
            MATCH (sn:StandardName {name_stage: 'accepted'})
            WHERE sn.source_paths IS NOT NULL AND size(sn.source_paths) > 0
            OPTIONAL MATCH (imas:IMASNode)-[:HAS_STANDARD_NAME]->(sn)
            WITH sn, collect(DISTINCT imas.id) AS hsn_paths
            OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
            WHERE src.source_type = 'dd' AND src.source_id IS NOT NULL
            WITH sn, hsn_paths, collect(DISTINCT src.source_id) AS produced_paths
            WITH sn, [p IN hsn_paths WHERE p IS NOT NULL]
                     + [p IN produced_paths WHERE p IS NOT NULL] AS edge_paths
            WITH sn, [p IN sn.source_paths WHERE NOT (p IN edge_paths)] AS stale
            WHERE size(stale) > 0
            RETURN sn.id AS name, stale
            ORDER BY name
            LIMIT 25
            """
        )
        assert not rows, (
            f"{len(rows)} accepted StandardName(s) have stale source_paths "
            "entries not backed by any live HAS_STANDARD_NAME / PRODUCED_NAME "
            "edge: " + "; ".join(f"{r['name']} → {r['stale']}" for r in rows)
        )

    def test_dd_attachment_unit_agrees_with_name(self, gc):
        """No SN attaches to a DD path whose (known) unit canonically disagrees.

        A ``HAS_STANDARD_NAME`` edge from an IMASNode asserts the DD path is an
        instance of the standard name; where the DD path carries a unit, it
        must agree with the name's unit after canonicalisation (or via a
        recorded equivalence / DD-side unit bug in ``dd_unit_exceptions``). A
        real dimensional conflict — e.g. a power name attached to a poloidal
        flux path — means the SN is attached to the wrong DD path.

        Scoped to DD paths that HAVE a HAS_UNIT edge (both sides known): a DD
        path with no unit edge (dimensionless ratios, counts, ``turns``,
        ``magnetic_shear``) has ``dd_unit = None``, which ``units_agree`` never
        treats as agreeing by design — those are DD-completeness gaps owned by
        the DD build, not SN edge-integrity defects, so they are out of scope
        here. Uses ``units_agree`` so curated equivalences/bugs are honoured.
        """
        from imas_codex.units.dd_unit_exceptions import units_agree

        rows = gc.query(
            """
            MATCH (imas:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
            WHERE sn.unit IS NOT NULL
              AND NOT (sn.name_stage IN $terminal)
            MATCH (imas)-[:HAS_UNIT]->(du:Unit)
            RETURN sn.id AS name, sn.unit AS sn_unit,
                   imas.id AS dd_path, du.id AS dd_unit
            """,
            terminal=_TERMINAL_STAGES,
        )
        offenders = [
            r for r in rows if not units_agree(r["sn_unit"], r["dd_unit"], r["dd_path"])
        ]
        assert not offenders, (
            f"{len(offenders)} SN↔DD attachment(s) with disagreeing units: "
            + "; ".join(
                f"{r['name']} ({r['sn_unit']}) ↮ {r['dd_path']} ({r['dd_unit']})"
                for r in offenders[:25]
            )
        )
