"""Unit-integrity tests for StandardName ↔ DD-path unit agreement.

For every live StandardName the mismatch axis compares the SN's declared unit
against the unit each of its DD source paths declares, normalising both sides
through the single canonical authority
(:func:`imas_standard_names.canonical_unit`, via
:mod:`imas_codex.units.dd_unit_exceptions`) so ordering-only and spelling-only
differences collapse. Two curated exception classes are suppressed rather than
flagged (see ``imas_codex/units/dd_unit_exceptions.yaml``): DD-side unit bugs
(the DD path carries a wrong unit and the SN correctly overrides it) and unit
equivalences (two canonical forms that are the same physical unit).

The axis reads the SN's live ``HAS_STANDARD_NAME`` edges — the graph source of
truth — NOT the denormalised ``source_paths`` scalar (which strands stale paths
of pruned/refined-away mappings). Terminal ``superseded``/``exhausted``/
``contested`` names are excluded: they are dead (replaced via ``REFINED_FROM``)
and not subject to the live-unit invariant.
"""

import pytest

from imas_codex.units.dd_unit_exceptions import (
    canonical_or_none,
    dd_unit_bug_globs,
    units_agree,
)

pytestmark = pytest.mark.graph

# Terminal name stages that are dead (replaced/abandoned) and excluded from the
# live-unit invariant.
_TERMINAL_STAGES = ["superseded", "exhausted", "contested"]


def _query_sn_unit_vs_dd(graph_client):
    """Return per-(name, path) rows over live SN → DD-path unit links.

    Uses the live ``HAS_STANDARD_NAME`` edge (IMASNode → StandardName), not the
    ``source_paths`` scalar, so a stale scalar path from a pruned mapping never
    reaches the comparison. One row per (SN, DD path); ``dd_units`` is the set
    of ``Unit`` ids on that path (normally one).
    """
    return graph_client.query(
        """
        MATCH (dd:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName {validation_status: 'valid'})
        WHERE sn.unit IS NOT NULL
          AND NOT coalesce(sn.name_stage, '') IN $terminal
        MATCH (dd)-[:HAS_UNIT]->(du:Unit)
        RETURN sn.id AS name, sn.unit AS sn_unit, dd.id AS path,
               collect(DISTINCT du.id) AS dd_units
        ORDER BY name, path
        """,
        terminal=_TERMINAL_STAGES,
    )


class TestSNUnitIntegrity:
    """StandardName unit must agree with its DD source path unit."""

    def test_sn_unit_matches_linked_dd_path_unit(self, graph_client):
        """Every live StandardName.unit must agree with its DD path unit.

        Both sides are normalised through ``canonical_unit`` so ordering- and
        spelling-only differences collapse. A curated DD-side unit bug (the DD
        path is wrong, the SN correct) or a recorded unit equivalence is
        suppressed via ``units_agree``. Skips cleanly if no SNs have DD-path
        unit linkage. DD-internal inconsistencies (a path carrying more than
        one Unit) are reported as skips, not failures.
        """
        rows = _query_sn_unit_vs_dd(graph_client)
        if not rows:
            pytest.skip("No live StandardNames with DD-path unit linkage")

        failures = []
        for r in rows:
            dd_units = r["dd_units"]

            # DD path carries more than one Unit → DD-internal issue, skip.
            if len(dd_units) > 1:
                continue

            # Canonical-equal, a recorded equivalence, or a curated DD-side
            # unit bug on this path → agree.
            if units_agree(r["sn_unit"], dd_units[0], r["path"]):
                continue

            failures.append(
                f"{r['name']}: SN unit={r['sn_unit']!r}, "
                f"DD unit={dd_units[0]!r} on {r['path']}"
            )

        assert not failures, (
            "StandardName units disagree with canonical DD units "
            f"({len(failures)} rows):\n  " + "\n  ".join(failures)
        )

    def test_declared_unit_has_matching_edge(self, graph_client):
        """A live StandardName that declares a unit must carry its HAS_UNIT edge.

        Scope:
        - ``unit IS NOT NULL`` — a name with no declared unit is a separate
          concern (e.g. an accepted name still awaiting a unit), not a
          property↔edge consistency failure.
        - excludes terminal ``superseded``/``exhausted``/``contested`` stages —
          dead names (replaced via ``REFINED_FROM`` or abandoned) whose unit
          edge may legitimately be gone; they are not subject to the live-unit
          invariant.
        """
        rows = graph_client.query(
            """
            MATCH (sn:StandardName {validation_status: 'valid'})
            WHERE sn.unit IS NOT NULL
              AND NOT coalesce(sn.name_stage, '') IN $terminal
              AND NOT (sn)-[:HAS_UNIT]->(:Unit)
            RETURN sn.id AS name, sn.unit AS unit
            ORDER BY name
            """,
            terminal=_TERMINAL_STAGES,
        )
        if not rows:
            return
        missing = [f"{r['name']} (unit={r['unit']!r})" for r in rows]
        assert not missing, (
            f"{len(missing)} live StandardNames declare a unit but lack the "
            "HAS_UNIT edge: " + ", ".join(missing[:20])
        )

    def test_sn_unit_property_matches_edge(self, graph_client):
        """A live SN's unit property must equal its HAS_UNIT target, canonically.

        The ``unit`` property is stored in the ISN canonical form while a
        ``Unit`` node id may have been written by the DD build's own formatter,
        so the two can differ by ordering/spelling only. Both are normalised
        through ``canonical_unit`` and a residual difference is a genuine
        property↔edge desync. Terminal stages are excluded (dead names).
        """
        rows = graph_client.query(
            """
            MATCH (sn:StandardName {validation_status: 'valid'})-[:HAS_UNIT]->(u:Unit)
            WHERE NOT coalesce(sn.name_stage, '') IN $terminal
            RETURN sn.id AS name, sn.unit AS prop_unit, u.id AS edge_unit
            ORDER BY name
            """,
            terminal=_TERMINAL_STAGES,
        )
        mismatches = [
            f"{r['name']}: property={r['prop_unit']!r}, edge={r['edge_unit']!r}"
            for r in rows
            if canonical_or_none(r["prop_unit"]) != canonical_or_none(r["edge_unit"])
        ]
        assert not mismatches, (
            "SN unit property disagrees with HAS_UNIT edge "
            f"({len(mismatches)} rows):\n  " + "\n  ".join(mismatches)
        )

    def test_no_imas_node_has_placeholder_unit(self, graph_client):
        """No IMASNode.unit may be a literal 'as_parent' placeholder.

        The DD uses ``as_parent`` / ``as_parent_level_2`` / ``as parent`` to
        mean "inherit the parent node's unit". Storing the literal placeholder
        corrupts the standard-name reviewer (it sees the placeholder instead
        of the real unit). Build-time resolution must eliminate these.
        """
        rows = graph_client.query("""
            MATCH (n:IMASNode)
            WHERE n.unit STARTS WITH 'as_parent' OR n.unit = 'as parent'
            RETURN n.id AS id, n.unit AS unit
            ORDER BY id
        """)
        if not rows:
            return
        sample = [f"{r['id']} [{r['unit']!r}]" for r in rows[:20]]
        assert not rows, (
            f"{len(rows)} IMASNodes carry an unresolved 'as_parent' unit "
            "placeholder:\n  " + "\n  ".join(sample)
        )

    def test_no_imas_node_has_multiple_units(self, graph_client):
        """No IMASNode may carry more than one HAS_UNIT edge.

        The DD build writes ``n.unit`` and a single ``HAS_UNIT`` edge. When a
        path's unit changes across a rebuild (a DD correction, or an
        as_parent placeholder resolving differently), the build self-heals by
        dropping existing HAS_UNIT edges before re-creating the canonical one.
        A node with two HAS_UNIT edges means a stale edge survived a unit
        change — the self-heal invariant is broken.
        """
        rows = graph_client.query("""
            MATCH (n:IMASNode)-[r:HAS_UNIT]->()
            WITH n, count(r) AS c
            WHERE c > 1
            RETURN n.id AS id, c AS edge_count
            ORDER BY id
        """)
        if not rows:
            return
        sample = [f"{r['id']} ({r['edge_count']} edges)" for r in rows[:20]]
        assert not rows, (
            f"{len(rows)} IMASNodes carry more than one HAS_UNIT edge "
            "(self-heal invariant broken):\n  " + "\n  ".join(sample)
        )

    def test_dd_side_unit_bug_globs_are_live(self, graph_client):
        """Every DD-side unit-bug glob must still match ≥1 live buggy DD path.

        Staleness guard on ``dd_unit_exceptions.yaml``: if a DD rebuild fixes a
        path's unit (or the path is removed), the glob stops matching any DD
        node that still carries the buggy unit, and the entry is dead residue
        that should be pruned. A glob that matches no DD ``Unit`` at all is
        surfaced as stale so the exception file does not accrete.

        Skips cleanly if the DD graph carries no ``Unit`` nodes at all.
        """
        import fnmatch

        globs = dd_unit_bug_globs()
        if not globs:
            pytest.skip("No DD-side unit-bug globs declared")

        dd_paths = graph_client.query(
            """
            MATCH (n:IMASNode)-[:HAS_UNIT]->(:Unit)
            RETURN n.id AS id
            """
        )
        if not dd_paths:
            pytest.skip("No IMASNodes carry a HAS_UNIT edge")
        ids = [r["id"] for r in dd_paths]

        stale = [g for g in globs if not any(fnmatch.fnmatchcase(i, g) for i in ids)]
        assert not stale, (
            "DD-side unit-bug globs match no live DD path (prune from "
            "dd_unit_exceptions.yaml):\n  " + "\n  ".join(stale)
        )
