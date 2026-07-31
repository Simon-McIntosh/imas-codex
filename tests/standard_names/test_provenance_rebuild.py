"""Tests for the one-time provenance rebuild (ledger fresh-parity recovery).

The rebuild replays the *deterministic* half of a fresh standard-names build
against the *existing* graph names: it rebinds ``StandardNameSource`` +
``PRODUCED_NAME`` (+ ``FROM_DD_PATH`` / ``HAS_PARENT``) so every live name
traces to >=1 source, WITHOUT regenerating names/docs. The authoritative
recovery map is an ISNC commit that still carried near-complete ``sources:``
blocks; the DD graph + ISN grammar close the remainder deterministically;
residue with no evidence remains unresolved for explicit investigation.

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

    assert set(recovered) == {"elongation_of_plasma_boundary", "plasma_current"}
    specs = recovered["elongation_of_plasma_boundary"]
    assert specs[0]["dd_path"] == "equilibrium/time_slice/boundary/elongation"
    assert specs[0]["source_type"] == "dd"
    assert recovered["plasma_current"] == [
        {
            "id": "catalog:plasma_current",
            "source_type": "catalog",
            "source_id": "plasma_current",
            "status": "attached",
        }
    ]


def test_load_recovery_map_missing_ref_returns_empty(tmp_path):
    """A non-existent ref yields an empty map rather than raising."""
    from imas_codex.standard_names.provenance_rebuild import load_recovery_map

    _init_isnc_repo(tmp_path, "- name: x\n  kind: scalar\n  unit: '1'\n")
    assert load_recovery_map(tmp_path, ref="deadbeef") == {}


def test_change_history_recovery_uses_existing_composed_sources():
    """Historical recovery selects real sources from the latest predecessor."""
    from imas_codex.standard_names.provenance_rebuild import (
        _fetch_change_history_sources,
    )

    gc = MagicMock()
    gc.query.side_effect = [
        [
            {
                "from_name": "intermediate_quantity",
                "to_name": "renamed_quantity",
                "changed_at": "2026-07-29T12:00:00Z",
            },
            {
                "from_name": "original_quantity",
                "to_name": "intermediate_quantity",
                "changed_at": "2026-07-28T12:00:00Z",
            },
        ],
        [
            {
                "source_id": "dd:path/b",
                "produced_sn_id": "original_quantity",
                "target_ids": [],
            },
            {
                "source_id": "dd:path/a",
                "produced_sn_id": None,
                "target_ids": ["original_quantity"],
            },
        ],
    ]

    recovered = _fetch_change_history_sources(gc, ["renamed_quantity"])

    assert recovered == {
        "renamed_quantity": ["dd:path/a", "dd:path/b"],
    }
    history_query = gc.query.call_args_list[0].args[0]
    source_query = gc.query.call_args_list[1].args[0]
    assert "change.changed_at DESC" in history_query
    assert "source.status IN ['composed', 'attached']" in source_query
    assert "live_target.name_stage" in source_query
    assert "scalar_target.id = source.produced_sn_id" in source_query
    assert "source.produced_sn_id IN $predecessor_ids" in source_query
    assert gc.query.call_args_list[0].kwargs["deletion_operations"]


# ---------------------------------------------------------------------------
# rebuild_provenance — orchestration/routing (mock-based)
# ---------------------------------------------------------------------------


def test_rebuild_routes_orphans_by_anchor_authority():
    """Remaining orphans are bound by descending anchor authority:
    recovery map (dd) > surviving source_paths scalar (dd) > change history.
    Childful parents use the structural reconciler; unsupported residue stays
    unresolved. No provenance source is fabricated.
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
        {"sn_id": "history_name", "name_stage": "accepted", "origin": "pipeline"},
        {"sn_id": "parent_a", "name_stage": "accepted", "origin": "derived"},
        {"sn_id": "parent_b", "name_stage": "accepted", "origin": "derived"},
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
        patch.object(
            pr,
            "find_orphan_parent_source_candidates",
            return_value=[
                {"parent_id": "parent_a", "dd_paths": []},
                {"parent_id": "parent_b", "dd_paths": []},
            ],
        ),
        patch.object(
            pr,
            "reconcile_orphan_parent_sources",
            return_value=2,
        ) as reconcile_parents,
        patch.object(pr, "_fetch_dd_source_paths", return_value=scalar_specs),
        patch.object(
            pr,
            "_fetch_change_history_sources",
            return_value={"history_name": ["dd:historical/path"]},
        ),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(
            pr,
            "bind_recovery_sources",
            side_effect=lambda name_id, specs, *, gc: bind_calls.append(
                (name_id, specs)
            ),
        ),
        patch.object(pr, "bind_sources_exclusively") as bind_history,
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map=recovery_map)

    assert m_fix.called  # fresh-build fixpoints (incl. seed_parent_sources) ran
    bound = dict(bind_calls)
    assert bound["in_map_name"][0]["source_type"] == "dd"
    assert bound["scalar_name"][0]["dd_path"] == "magnetics/flux_loop/area"
    bind_history.assert_called_once_with(gc, "history_name", ["dd:historical/path"])
    assert "residue_name" not in bound

    assert summary["bound_from_map"] == 1
    assert summary["bound_from_scalar"] == 1
    assert summary["bound_from_history"] == 1
    assert summary["history_recoverable_names"] == ["history_name"]
    assert summary["parent_source_candidates"] == 2
    assert summary["parent_sources_reconciled"] == 2
    assert summary["unresolved"] == 1
    shared_classification = reconcile_parents.call_args.kwargs["classification"]
    assert [row["parent_id"] for row in shared_classification["repairable"]] == [
        "parent_a",
        "parent_b",
    ]


def test_rejected_derived_parent_remains_in_unresolved_classification():
    """Admission rejection must not hide a derived orphan from fallback."""
    import imas_codex.standard_names.provenance_rebuild as pr
    from imas_codex.standard_names.parents import AdmissionResult

    parent_id = "line_integrated_impurity_ion_velocity"
    orphans = [
        {
            "sn_id": parent_id,
            "name_stage": "pending",
            "origin": "derived",
        }
    ]
    gc = MagicMock()

    with (
        patch.object(pr, "find_provenance_orphans", return_value=orphans),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(
            pr,
            "find_orphan_parent_source_candidates",
            return_value=[
                {
                    "parent_id": parent_id,
                    "origin": "derived",
                    "dd_paths": [
                        "spectrometer_x_ray_crystal/channel/"
                        "profiles_line_integrated/velocity_tor"
                    ],
                }
            ],
        ),
        patch(
            "imas_codex.standard_names.parents.is_admissible_parent_name",
            return_value=AdmissionResult(
                admit=False,
                reason="suppressed: single-child shadow",
                clause=None,
            ),
        ),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_change_history_sources", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={}, dry_run=True)

    assert summary["parent_source_candidates"] == 0
    assert summary["parent_source_rejected_names"] == [parent_id]
    assert summary["unresolved_names"] == [parent_id]


def test_non_derived_parent_bypasses_structural_admission_gate() -> None:
    """Pipeline refinement parents retain structural provenance recovery."""
    from imas_codex.standard_names.graph_ops import (
        reconcile_orphan_parent_sources,
    )
    from imas_codex.standard_names.parents import AdmissionResult

    parent_id = "line_integrated_impurity_ion_velocity"
    gc = MagicMock()
    gc.query.return_value = []
    with (
        patch(
            "imas_codex.standard_names.graph_ops.find_orphan_parent_source_candidates",
            return_value=[
                {
                    "parent_id": parent_id,
                    "origin": "pipeline",
                    "dd_paths": [],
                }
            ],
        ),
        patch(
            "imas_codex.standard_names.parents.is_admissible_parent_name",
            return_value=AdmissionResult(
                admit=False,
                reason="would be rejected if structural admission applied",
                clause=None,
            ),
        ) as admission,
    ):
        seeded = reconcile_orphan_parent_sources(gc=gc)

    assert seeded == 1
    admission.assert_not_called()
    write = next(
        item
        for item in gc.query.call_args_list
        if "MERGE (sns:StandardNameSource" in item.args[0]
    )
    assert write.kwargs["parent_id"] == parent_id


def test_parent_source_reconcile_keeps_dd_identity_with_the_child() -> None:
    """Structural sources stay distinct even when every parent shares one leaf."""
    from imas_codex.standard_names.graph_ops import (
        reconcile_orphan_parent_sources,
    )
    from imas_codex.standard_names.parents import AdmissionResult

    leaf = "spectrometer_visible/channel/processed_line/radiance"
    parent_ids = ["photon_radiance", "radiance"]
    candidates = [
        {"parent_id": parent_id, "origin": "derived", "dd_paths": [leaf]}
        for parent_id in parent_ids
    ]
    gc = MagicMock()
    gc.query.side_effect = [
        candidates,
        [],
        [],
        [],
    ]

    with patch(
        "imas_codex.standard_names.parents.is_admissible_parent_name",
        return_value=AdmissionResult(admit=True, reason="valid", clause=None),
    ):
        assert reconcile_orphan_parent_sources(gc=gc) == 2
        assert reconcile_orphan_parent_sources(gc=gc) == 0

    writes = [
        item
        for item in gc.query.call_args_list
        if "MERGE (sns:StandardNameSource" in item.args[0]
    ]
    assert [item.kwargs["source_node_id"] for item in writes] == [
        "derived:photon_radiance",
        "derived:radiance",
    ]
    assert [item.kwargs["source_type"] for item in writes] == [
        "derived",
        "derived",
    ]
    assert [item.kwargs["source_id"] for item in writes] == parent_ids
    assert all(item.kwargs["source_node_id"] != f"dd:{leaf}" for item in writes)

    assert all(
        "MERGE (sns)-[:FROM_DD_PATH]" not in item.args[0]
        for item in gc.query.call_args_list
    )


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
        patch.object(
            pr,
            "find_orphan_parent_source_candidates",
            return_value=[{"parent_id": "x", "dd_paths": []}],
        ),
        patch.object(pr, "reconcile_orphan_parent_sources") as m_parents,
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_change_history_sources", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(pr, "bind_recovery_sources") as m_bind,
        patch.object(pr, "bind_sources_exclusively") as m_bind_history,
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={}, dry_run=True)

    assert not m_re.called  # dry run mutates nothing
    assert not m_fix.called
    assert not m_parents.called
    assert summary["parent_source_candidates"] == 1
    assert summary["parent_source_candidate_names"] == ["x"]
    assert summary["parent_sources_reconciled"] == 0
    assert not m_bind.called
    assert not m_bind_history.called
    assert summary["dry_run"] is True
    assert summary["unresolved"] == 0


def test_rebuild_excludes_reattachable_desyncs_from_fallback_binding():
    """A name orphaned only by a missing edge (its source still names it via
    produced_sn_id) is reattached to its TRUE source, never bound to a fresh
    unsupported fallback.
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
        patch.object(pr, "find_orphan_parent_source_candidates", return_value=[]),
        patch.object(pr, "reconcile_orphan_parent_sources", return_value=0),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_change_history_sources", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(
            pr,
            "bind_recovery_sources",
            side_effect=lambda name_id, specs, *, gc: bind_calls.append(name_id),
        ),
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={})

    assert m_re.called
    assert "desync_name" not in bind_calls
    assert bind_calls == []
    assert summary["reattached"] == 1
    assert summary["unresolved"] == 1


def test_rebuild_excludes_pending_source_names_from_fallback():
    """A live orphan whose real dd source is still PENDING (extracted) in the
    GENERATE_NAME queue is left for the pipeline — never given an unsupported
    fallback that would pin fabricated provenance over the real source about
    to be composed.
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
        patch.object(pr, "find_orphan_parent_source_candidates", return_value=[]),
        patch.object(pr, "reconcile_orphan_parent_sources", return_value=0),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_change_history_sources", return_value={}),
        # pending_name has a claimable pending extracted source in the queue
        patch.object(pr, "_fetch_pending_source_names", return_value={"pending_name"}),
        patch.object(
            pr,
            "bind_recovery_sources",
            side_effect=lambda name_id, specs, *, gc: bind_calls.append(name_id),
        ),
    ):
        summary = pr.rebuild_provenance(gc=gc, recovery_map={})

    # the pending-source name is excluded from ANY fallback binding
    assert "pending_name" not in bind_calls
    assert bind_calls == []
    assert summary["excluded_pending"] == 1
    assert summary["unresolved"] == 1


def test_rebuild_can_retire_explicit_unrecoverable_residue() -> None:
    import imas_codex.standard_names.provenance_rebuild as pr

    orphans = [{"sn_id": "lost_name", "name_stage": "accepted", "origin": "pipeline"}]
    gc = MagicMock()

    with (
        patch.object(
            pr,
            "find_provenance_orphans",
            side_effect=[orphans, orphans, orphans, []],
        ),
        patch.object(pr, "find_edge_scalar_desyncs", return_value=[]),
        patch.object(pr, "reattach_produced_name_edges", return_value=0),
        patch.object(pr, "_run_deterministic_fixpoints"),
        patch.object(pr, "find_orphan_parent_source_candidates", return_value=[]),
        patch.object(pr, "reconcile_orphan_parent_sources", return_value=0),
        patch.object(pr, "_fetch_dd_source_paths", return_value={}),
        patch.object(pr, "_fetch_change_history_sources", return_value={}),
        patch.object(pr, "_fetch_pending_source_names", return_value=set()),
        patch.object(
            pr,
            "retire_unrecoverable_provenance_orphans",
            return_value=["lost_name"],
        ) as retire,
    ):
        summary = pr.rebuild_provenance(
            gc=gc,
            recovery_map={},
            retire_unresolved=True,
            include_accepted_retirement=True,
        )

    retire.assert_called_once_with(gc, ["lost_name"], include_accepted=True)
    assert summary["retired_unresolved"] == 1
    assert summary["retired_unresolved_names"] == ["lost_name"]
    assert summary["orphans_after"] == 0
    assert summary["unresolved"] == 0
