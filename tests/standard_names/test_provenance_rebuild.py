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


def test_rebuild_routes_orphans_by_anchor_authority():
    """Remaining orphans are bound by descending anchor authority:
    recovery map (dd) > surviving source_paths scalar (dd) > derived parent
    (composed-from-children) > manual residue. Never a fabricated DD path.
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
        {"sn_id": "scalar_name", "name_stage": "accepted", "origin": "catalog_edit"},
        {"sn_id": "parent_name", "name_stage": "accepted", "origin": "catalog_edit"},
        {"sn_id": "residue_name", "name_stage": "accepted", "origin": "catalog_edit"},
    ]
    scalar_specs = {
        "scalar_name": [
            {
                "id": "dd:magnetics/flux_loop/area",
                "source_type": "dd",
                "dd_path": "magnetics/flux_loop/area",
                "status": "attached",
            }
        ]
    }
    gc = MagicMock()
    bind_calls = []

    with (
        patch.object(pr, "find_provenance_orphans", return_value=orphans),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(pr, "reattach_produced_name_edges", return_value=0),
        patch.object(pr, "_run_deterministic_fixpoints") as m_fix,
        patch.object(pr, "_fetch_dd_source_paths", return_value=scalar_specs),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(pr, "_classify_derived_parents", return_value={"parent_name"}),
        patch.object(
            pr,
            "bind_recovery_sources",
            side_effect=lambda name_id, specs, *, gc: bind_calls.append(
                (name_id, specs)
            ),
        ),
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map=recovery_map)

    assert m_fix.called  # fresh-build fixpoints (incl. seed_parent_sources) ran
    bound = dict(bind_calls)
    assert bound["in_map_name"][0]["source_type"] == "dd"
    assert bound["scalar_name"][0]["dd_path"] == "magnetics/flux_loop/area"
    # a derived PARENT gets a derived source (composed-from-children), not manual
    assert bound["parent_name"][0]["source_type"] == "derived"
    assert bound["parent_name"][0]["id"] == "derived:parent_name"
    # only genuine residue with no anchor becomes manual, never a fabricated dd
    assert bound["residue_name"][0]["source_type"] == "manual"
    assert "dd_path" not in bound["residue_name"][0]

    assert summary["bound_from_map"] == 1
    assert summary["bound_from_scalar"] == 1
    assert summary["bound_derived"] == 1
    assert summary["bound_manual"] == 1


def test_rebuild_dry_run_binds_nothing():
    """A dry run classifies but performs no writes and no fixpoints."""
    import imas_codex.standard_names.provenance_rebuild as pr

    orphans = [{"sn_id": "x", "name_stage": "accepted", "origin": "catalog_edit"}]
    gc = MagicMock()

    with (
        patch.object(pr, "find_provenance_orphans", return_value=orphans),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(pr, "reattach_produced_name_edges") as m_re,
        patch.object(pr, "_run_deterministic_fixpoints") as m_fix,
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(pr, "_classify_derived_parents", return_value=set()),
        patch.object(pr, "bind_recovery_sources") as m_bind,
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={}, dry_run=True)

    assert not m_re.called  # dry run mutates nothing
    assert not m_fix.called
    assert not m_bind.called
    assert summary["dry_run"] is True
    assert summary["bound_manual"] == 1  # would-be classification still reported


def test_rebuild_excludes_reattachable_desyncs_from_fallback_binding():
    """A name orphaned only by a missing edge (its source still names it via
    produced_sn_id) is reattached to its TRUE source, never bound to a fresh
    manual/derived fallback.
    """
    import imas_codex.standard_names.provenance_rebuild as pr

    orphans = [
        {"sn_id": "desync_name", "name_stage": "accepted", "origin": "pipeline"},
        {"sn_id": "residue_name", "name_stage": "accepted", "origin": "catalog_edit"},
    ]
    desyncs = [{"source_id": "dd:x", "sn_id": "desync_name", "name_stage": "accepted"}]
    gc = MagicMock()
    bind_calls = []

    with (
        patch.object(pr, "find_provenance_orphans", return_value=orphans),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=desyncs),
        patch.object(pr, "reattach_produced_name_edges", return_value=1) as m_re,
        patch.object(pr, "_run_deterministic_fixpoints"),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(pr, "_classify_derived_parents", return_value=set()),
        patch.object(
            pr,
            "bind_recovery_sources",
            side_effect=lambda name_id, specs, *, gc: bind_calls.append(name_id),
        ),
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={})

    assert m_re.called
    assert "desync_name" not in bind_calls
    assert bind_calls == ["residue_name"]
    assert summary["reattached"] == 1


def test_rebuild_excludes_pending_source_names_from_fallback():
    """A live orphan whose real dd source is still PENDING (extracted) in the
    GENERATE_NAME queue is left for the pipeline — never given a synthesized
    manual/derived fallback that would pin a fabricated source over the real
    one about to be composed.
    """
    import imas_codex.standard_names.provenance_rebuild as pr

    orphans = [
        {"sn_id": "pending_name", "name_stage": "accepted", "origin": "catalog_edit"},
        {"sn_id": "residue_name", "name_stage": "accepted", "origin": "catalog_edit"},
    ]
    gc = MagicMock()
    bind_calls = []

    with (
        patch.object(pr, "find_provenance_orphans", return_value=orphans),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(pr, "reattach_produced_name_edges", return_value=0),
        patch.object(pr, "_run_deterministic_fixpoints"),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        # pending_name has a claimable pending extracted source in the queue
        patch.object(pr, "_fetch_pending_source_names", return_value={"pending_name"}),
        patch.object(pr, "_classify_derived_parents", return_value=set()),
        patch.object(
            pr,
            "bind_recovery_sources",
            side_effect=lambda name_id, specs, *, gc: bind_calls.append(name_id),
        ),
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={})

    # the pending-source name is excluded from ANY fallback binding
    assert "pending_name" not in bind_calls
    assert bind_calls == ["residue_name"]
    assert summary["excluded_pending"] == 1
    assert summary["bound_manual"] == 1
    assert summary["bound_derived"] == 0
