"""Tests for the one-time provenance rebuild (ledger fresh-parity recovery).

The rebuild replays the *deterministic* half of a fresh standard-names build
against the *existing* graph names: it rebinds ``StandardNameSource`` +
``PRODUCED_NAME`` (+ ``FROM_DD_PATH`` / ``HAS_PARENT``) so every live name
traces to >=1 source, WITHOUT regenerating names/docs. The authoritative
recovery map is an ISNC commit that still carried near-complete ``sources:``
blocks; the DD graph + ISN grammar close the remainder deterministically;
residue with no anchor becomes an explicit ``source_type='manual'`` source.

These tests are mock-based (no live Neo4j) except where marked ``graph`` —
per repo convention, mutation logic is asserted against captured Cypher, and
only read-only ledger invariants run against the live graph.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _captured_queries(gc: MagicMock) -> list[str]:
    """All Cypher strings passed to ``gc.query`` on an injected mock client."""
    out: list[str] = []
    for call in gc.query.call_args_list:
        out.append(call.args[0] if call.args else call.kwargs["query"])
    return out


def _last_query_params(gc: MagicMock) -> dict:
    return dict(gc.query.call_args.kwargs)


# ---------------------------------------------------------------------------
# recovery_sources_from_entries — pure parse of the ISNC recovery map
# ---------------------------------------------------------------------------


def test_recovery_map_extracts_dd_sources_keyed_by_name():
    """A catalog entry's ``sources:`` block is recovered, keyed by name.

    The a2f8831 format is a list of ``{id, dd_path, status}`` per entry. We
    key by the entry ``name`` and normalise each source to the fields the
    rebuild needs to reconstruct a StandardNameSource.
    """
    from imas_codex.standard_names.provenance_rebuild import (
        recovery_sources_from_entries,
    )

    entries = [
        {
            "name": "elongation_of_plasma_boundary",
            "sources": [
                {
                    "id": "dd:equilibrium/time_slice/boundary/elongation",
                    "dd_path": "equilibrium/time_slice/boundary/elongation",
                    "status": "attached",
                },
                {
                    "id": "dd:summary/boundary/elongation/value",
                    "dd_path": "summary/boundary/elongation/value",
                    "status": "attached",
                },
            ],
        },
        # An entry with no sources block contributes nothing.
        {"name": "plasma_current"},
    ]

    recovered = recovery_sources_from_entries(entries)

    assert set(recovered) == {"elongation_of_plasma_boundary"}
    specs = recovered["elongation_of_plasma_boundary"]
    assert len(specs) == 2
    first = specs[0]
    assert first["id"] == "dd:equilibrium/time_slice/boundary/elongation"
    assert first["source_type"] == "dd"
    assert first["dd_path"] == "equilibrium/time_slice/boundary/elongation"


# ---------------------------------------------------------------------------
# bind_recovery_sources — MERGE StandardNameSource + PRODUCED_NAME + FROM_DD_PATH
# ---------------------------------------------------------------------------


def test_bind_recovery_sources_merges_source_edge_and_dd_path():
    """Binding a dd source MERGEs the source, links PRODUCED_NAME, mirrors
    ``produced_sn_id``, and links FROM_DD_PATH — all gated on the name existing.
    """
    from imas_codex.standard_names.provenance_rebuild import bind_recovery_sources

    specs = [
        {
            "id": "dd:equilibrium/time_slice/boundary/elongation",
            "source_type": "dd",
            "dd_path": "equilibrium/time_slice/boundary/elongation",
            "status": "attached",
        }
    ]
    gc = MagicMock()
    gc.query.return_value = [{"bound": 1}]
    bound = bind_recovery_sources("elongation_of_plasma_boundary", specs, gc=gc)

    assert bound == 1
    cypher = _captured_queries(gc)[0]
    flat = " ".join(cypher.split())
    # gate: the SN MATCH precedes the source status SET
    assert flat.find("MATCH (sn:StandardName") != -1
    assert flat.find("MATCH (sn:StandardName") < flat.find("SET sns.source_type")
    assert "MERGE (sns:StandardNameSource" in flat
    assert "PRODUCED_NAME" in flat
    assert "produced_sn_id" in flat
    assert "FROM_DD_PATH" in flat
    # the specs are passed as a bound parameter (UNWIND-friendly), not inlined
    params = _last_query_params(gc)
    assert params.get("name_id") == "elongation_of_plasma_boundary"
    assert params.get("specs") == specs


# ---------------------------------------------------------------------------
# load_recovery_map — extract the sources map from a catalog commit via git
# ---------------------------------------------------------------------------


def _init_isnc_repo(root: Path, domain_yaml: str) -> None:
    """Create a minimal ISNC git repo with one committed domain file."""
    (root / "standard_names").mkdir(parents=True)
    (root / "standard_names" / "equilibrium.yml").write_text(domain_yaml)
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    for args in (
        ["git", "init", "-q"],
        ["git", "add", "standard_names/equilibrium.yml"],
        ["git", "commit", "-q", "-m", "seed"],
    ):
        subprocess.run(args, cwd=root, check=True, env={**env, "HOME": str(root)})


def test_load_recovery_map_reads_sources_from_a_git_ref(tmp_path):
    """load_recovery_map extracts the ``sources:`` map from the YAML tree at a
    specific commit (not the working tree), keyed by name.
    """
    from imas_codex.standard_names.provenance_rebuild import load_recovery_map

    domain = (
        "- name: elongation_of_plasma_boundary\n"
        "  kind: scalar\n"
        "  unit: '1'\n"
        "  sources:\n"
        "  - id: dd:equilibrium/time_slice/boundary/elongation\n"
        "    dd_path: equilibrium/time_slice/boundary/elongation\n"
        "    status: attached\n"
        "- name: plasma_current\n"
        "  kind: scalar\n"
        "  unit: A\n"
    )
    _init_isnc_repo(tmp_path, domain)

    recovered = load_recovery_map(tmp_path, ref="HEAD")

    assert set(recovered) == {"elongation_of_plasma_boundary"}
    specs = recovered["elongation_of_plasma_boundary"]
    assert specs[0]["dd_path"] == "equilibrium/time_slice/boundary/elongation"
    assert specs[0]["source_type"] == "dd"


def test_load_recovery_map_missing_ref_returns_empty(tmp_path):
    """A non-existent ref yields an empty map rather than raising."""
    from imas_codex.standard_names.provenance_rebuild import load_recovery_map

    _init_isnc_repo(tmp_path, "- name: x\n  kind: scalar\n  unit: '1'\n")
    assert load_recovery_map(tmp_path, ref="deadbeef") == {}


# ---------------------------------------------------------------------------
# rebuild_provenance — orchestration/routing (mock-based)
# ---------------------------------------------------------------------------


def test_rebuild_routes_orphans_to_map_derived_and_manual():
    """Each orphan is bound by the conservative decision tree:

    - in the recovery map      → its dd/signal sources
    - else grammar-derived      → an explicit ``derived`` source
    - else residue (no anchor)  → an explicit ``manual`` source (never a
      fabricated DD path)

    Then the structural/DD fixpoints run to reach fresh-parity.
    """
    import imas_codex.standard_names.provenance_rebuild as pr

    recovery_map = {
        "in_map_name": [
            {
                "id": "dd:equilibrium/time_slice/boundary/elongation",
                "source_type": "dd",
                "dd_path": "equilibrium/time_slice/boundary/elongation",
                "status": "attached",
            }
        ]
    }
    orphans = [
        {"sn_id": "in_map_name", "name_stage": "accepted", "origin": "catalog_edit"},
        {"sn_id": "derived_name", "name_stage": "accepted", "origin": "derived"},
        {"sn_id": "residue_name", "name_stage": "accepted", "origin": "catalog_edit"},
    ]
    gc = MagicMock()
    # classification query → derived_name has a parent/derived, residue_name does not
    gc.query.return_value = [
        {"id": "derived_name", "derived": True},
        {"id": "residue_name", "derived": False},
    ]

    bind_calls = []

    def _spy_bind(name_id, specs, *, gc):  # noqa: ANN001
        bind_calls.append((name_id, specs))
        return len(specs)

    with (
        patch.object(pr, "find_provenance_orphans", return_value=orphans),
        patch.object(pr, "bind_recovery_sources", side_effect=_spy_bind),
        patch.object(pr, "rederive_structural_edges", return_value={}) as m_struct,
        patch.object(pr, "reconcile_standard_name_sources", return_value={}) as m_recon,
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map=recovery_map)

    bound = dict(bind_calls)
    # in-map name bound from the recovery map (dd source)
    assert bound["in_map_name"][0]["source_type"] == "dd"
    # derived name bound with a synthesised derived source
    assert bound["derived_name"][0]["source_type"] == "derived"
    assert bound["derived_name"][0]["id"] == "derived:derived_name"
    # residue bound with an explicit manual source, never a fabricated dd path
    assert bound["residue_name"][0]["source_type"] == "manual"
    assert bound["residue_name"][0]["id"] == "manual:residue_name"
    assert "dd_path" not in bound["residue_name"][0]

    # fresh-parity fixpoints ran
    assert m_struct.called
    assert m_recon.called

    assert summary["orphans_before"] == 3
    assert summary["bound_from_map"] == 1
    assert summary["bound_derived"] == 1
    assert summary["bound_manual"] == 1


def test_rebuild_dry_run_binds_nothing():
    """A dry run classifies but performs no writes and no fixpoints."""
    import imas_codex.standard_names.provenance_rebuild as pr

    orphans = [{"sn_id": "x", "name_stage": "accepted", "origin": "catalog_edit"}]
    gc = MagicMock()
    gc.query.return_value = [{"id": "x", "derived": False}]

    with (
        patch.object(pr, "find_provenance_orphans", return_value=orphans),
        patch.object(pr, "bind_recovery_sources") as m_bind,
        patch.object(pr, "rederive_structural_edges") as m_struct,
        patch.object(pr, "reconcile_standard_name_sources") as m_recon,
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={}, dry_run=True)

    assert not m_bind.called
    assert not m_struct.called
    assert not m_recon.called
    assert summary["dry_run"] is True
    assert summary["bound_manual"] == 1  # would-be classification still reported
