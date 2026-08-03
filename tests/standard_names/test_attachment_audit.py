"""Retroactive re-validation of source→name attachments already in the graph.

The consistency guard is evaluated at compose time only, so every attachment
written before a rule existed — or written by one of the paths that never
consults it (refine successor migration, edit rebind, derived-parent seeding,
provenance rebuild) — is permanently wrong. These tests cover the reconcile
that re-asks the guard's question of the graph: what it detaches, what it
protects, where the freed source goes, and that a second pass is a no-op.

Two properties get their own sections because they decide whether the reconcile
is safe to run at all: the guard's one order-DEPENDENT rule must be applied with
compose semantics (a conflicting group keeps one representative, deterministically
chosen), and a name whose every source is rejected by one rule is a NAME defect
for ``sn edit --rename``, never an attachment defect to detach.

Mocked unit tests run in the default tier; the end-to-end guarantees run
against a live graph (``@pytest.mark.graph``).
"""

from __future__ import annotations

import json
import os
import uuid
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import TracebackType
from typing import Any
from unittest.mock import MagicMock

import pytest

from imas_codex.standard_names.attachment_audit import (
    AttachmentAuditResult,
    AttachmentVerdict,
    audit_attachments,
    reconcile_attachment_consistency,
)

_FIXTURE_ID_STEM = "test_attach_audit__"


# ---------------------------------------------------------------------------
# Mocked — verdict classification and the audit predicate
# ---------------------------------------------------------------------------


def _row(
    dd_path: str,
    sn_id: str,
    *,
    name_stage: str = "drafted",
    dd_unit: str | None = None,
    sn_unit: str | None = None,
    other_live_names: int = 0,
) -> dict:
    return {
        "source_node_id": f"dd:{dd_path}",
        "dd_path": dd_path,
        "sn_id": sn_id,
        "name_stage": name_stage,
        "origin": "pipeline",
        "dd_unit": dd_unit,
        "sn_unit": sn_unit,
        "other_live_names": other_live_names,
    }


def _client(rows: list[dict]) -> MagicMock:
    """A client that answers the audit read with *rows* and writes with a count."""
    from imas_codex.standard_names import attachment_audit as mod

    def _query(q: str, **params):
        # Matched on a distinctive projection rather than by identity: the read is
        # a template whose scope clause is filled in per call (whole corpus vs one
        # name), so the query string reaching the client is never the raw constant.
        if "AS other_live_names" in q:
            if (want := params.get("sn_id")) is not None:
                return [r for r in rows if r["sn_id"] == want]
            return rows
        if q == mod._DETACH_QUERY:
            return [{"detached": len(params.get("items") or [])}]
        return []

    gc = MagicMock()
    gc.query.side_effect = _query
    return gc


def test_audit_accepts_consistent_attachment() -> None:
    gc = _client(
        [
            _row(
                "core_profiles/profiles_1d/electrons/density",
                "electron_density",
                dd_unit="m^-3",
                sn_unit="m^-3",
            )
        ]
    )
    result = audit_attachments(gc)
    assert result.checked == 1
    assert result.rejected == []


def test_audit_rejects_locus_device_mismatch() -> None:
    """The live defect: a strike-point source on a camera-orientation name."""
    gc = _client(
        [
            _row(
                "summary/boundary/strike_point_inner_z/value",
                "z_image_up_unit_vector_of_camera",
            )
        ]
    )
    result = audit_attachments(gc)
    assert len(result.rejected) == 1
    v = result.rejected[0]
    assert v.rule == "locus/source device mismatch"
    assert v.dd_path == "summary/boundary/strike_point_inner_z/value"


def test_audit_rejects_unit_dimensionality() -> None:
    gc = _client(
        [
            _row(
                "equilibrium/time_slice/profiles_1d/gm3",
                "radial_flux_surface_averaged_metric",
                dd_unit="1",
                sn_unit="m^-2",
            )
        ]
    )
    result = audit_attachments(gc)
    assert len(result.rejected) == 1
    assert result.rejected[0].rule == "unit dimensionality mismatch"


def test_audit_passes_registry_excepted_dd_defect() -> None:
    """A recorded DD-side unit bug is not an attachment defect."""
    gc = _client(
        [
            _row(
                "camera_ir/channel/camera/up/z",
                "z_image_up_unit_vector_of_camera",
                dd_unit="m",
                sn_unit="1",
            )
        ]
    )
    assert audit_attachments(gc).rejected == []


# ---------------------------------------------------------------------------
# Mocked — order-dependent rules carry compose semantics
# ---------------------------------------------------------------------------

_CAMERA = "camera_ir/channel/camera"
_VECTOR_NAME = "z_direction_unit_vector_of_camera"


def test_conflicting_group_keeps_one_representative() -> None:
    """The distinct-vector rule is pairwise: one member of a group survives.

    At compose time the conflicting sources arrive one at a time and only the
    ACCEPTED ones accumulate, so the first is kept and the rest rejected. The
    audit must reproduce that, not reject the whole group.
    """
    gc = _client(
        [
            _row(f"{_CAMERA}/direction/z", _VECTOR_NAME),
            _row(f"{_CAMERA}/up/z", _VECTOR_NAME),
        ]
    )
    result = audit_attachments(gc)
    assert result.checked == 2
    assert len(result.rejected) == 1, "a two-way conflict is one surplus, not two"
    assert result.rejected[0].rule == "distinct-vector conflict"
    assert result.rejected[0].dd_path == f"{_CAMERA}/up/z"


def test_conflicting_group_verdict_is_deterministic() -> None:
    """Row order is Neo4j's; two runs must reject the same member."""
    rows = [
        _row(f"{_CAMERA}/up/z", _VECTOR_NAME),
        _row(f"{_CAMERA}/direction/z", _VECTOR_NAME),
    ]
    forward = audit_attachments(_client(list(rows)))
    reverse = audit_attachments(_client(list(reversed(rows))))
    assert [v.dd_path for v in forward.rejected] == [f"{_CAMERA}/up/z"]
    assert [v.dd_path for v in reverse.rejected] == [f"{_CAMERA}/up/z"]


def test_three_way_conflict_rejects_only_the_surplus() -> None:
    """N mutually-conflicting attachments are N-1 rejections."""
    gc = _client(
        [
            _row(f"{_CAMERA}/up/z", _VECTOR_NAME),
            _row(f"{_CAMERA}/line_of_sight/z", _VECTOR_NAME),
            _row(f"{_CAMERA}/direction/z", _VECTOR_NAME),
        ]
    )
    result = audit_attachments(gc)
    assert {v.dd_path for v in result.rejected} == {
        f"{_CAMERA}/line_of_sight/z",
        f"{_CAMERA}/up/z",
    }


def test_accumulation_does_not_alter_an_order_independent_verdict() -> None:
    """Tense / state / locus / unit rules look only at the one attachment.

    Accumulating accepted siblings for the pairwise rule must not make an
    order-independent verdict depend on where the attachment sits in the group.
    """
    good = "core_profiles/profiles_1d/electrons/density"
    bad = "core_profiles/profiles_1d/electrons/state/density"
    name = "electron_density"
    for rows in (
        [_row(good, name), _row(bad, name)],
        [_row(bad, name), _row(good, name)],
    ):
        result = audit_attachments(_client(rows))
        assert [v.dd_path for v in result.rejected] == [bad]
        assert result.rejected[0].rule == "state-resolution mismatch"


def test_by_rule_groups_rejections() -> None:
    result = AttachmentAuditResult(checked=3)
    result.rejected = [
        AttachmentVerdict("s1", "p1", "n1", "drafted", "tense mismatch: a"),
        AttachmentVerdict("s2", "p2", "n2", "drafted", "tense mismatch: b"),
        AttachmentVerdict(
            "s3", "p3", "n3", "drafted", "unit dimensionality mismatch: c"
        ),
    ]
    assert result.by_rule() == {"tense mismatch": 2, "unit dimensionality mismatch": 1}


def test_accepted_stage_is_protected() -> None:
    assert AttachmentVerdict("s", "p", "n", "accepted", "r").protected
    assert not AttachmentVerdict("s", "p", "n", "drafted", "r").protected
    assert not AttachmentVerdict("s", "p", "n", None, "r").protected


def test_superseded_stage_is_historical() -> None:
    """A supersede leaves its edges intact on purpose — they are provenance."""
    assert AttachmentVerdict("s", "p", "n", "superseded", "r").historical
    assert not AttachmentVerdict("s", "p", "n", "exhausted", "r").historical


def test_reroute_only_when_no_live_name_remains() -> None:
    """A source still backing another live name must not be rewound."""
    assert AttachmentVerdict("s", "p", "n", "exhausted", "r", 0).reroute
    assert not AttachmentVerdict("s", "p", "n", "exhausted", "r", 1).reroute


def test_superseded_attachment_is_never_detached() -> None:
    gc = _client(
        [
            _row(
                "summary/boundary/strike_point_inner_z/value",
                "z_image_up_unit_vector_of_camera",
                name_stage="superseded",
            )
        ]
    )
    result = reconcile_attachment_consistency(gc, include_accepted=True)
    assert len(result.rejected) == 1
    assert result.skipped_historical == 1
    assert result.detached == 0
    assert gc.query.call_count == 1


def test_source_with_another_live_name_is_not_rerouted() -> None:
    """The stale edge goes; the source keeps the good name it already produced."""
    gc = _client(
        [
            _row(
                "summary/boundary/strike_point_inner_z/value",
                "z_image_up_unit_vector_of_camera",
                name_stage="exhausted",
                other_live_names=1,
            )
        ]
    )
    result = reconcile_attachment_consistency(gc)
    assert len(result.rejected) == 1
    assert result.sources_rerouted == 0
    items = gc.query.call_args_list[1].kwargs["items"]
    assert items[0]["reroute"] is False


def test_dry_run_writes_nothing() -> None:
    gc = _client(
        [
            _row(
                "summary/boundary/strike_point_inner_z/value",
                "z_image_up_unit_vector_of_camera",
            )
        ]
    )
    result = reconcile_attachment_consistency(gc, dry_run=True)
    assert len(result.rejected) == 1
    assert result.detached == 0
    # Only the read query ran.
    assert gc.query.call_count == 1


def test_accepted_names_are_not_detached_by_default() -> None:
    gc = _client(
        [
            _row(
                "summary/boundary/strike_point_inner_z/value",
                "z_image_up_unit_vector_of_camera",
                name_stage="accepted",
            )
        ]
    )
    result = reconcile_attachment_consistency(gc)
    assert len(result.rejected) == 1
    assert result.skipped_protected == 1
    assert result.detached == 0
    assert gc.query.call_count == 1


# ---------------------------------------------------------------------------
# Mocked — a whole-name wipeout is a NAME defect, not an attachment defect
# ---------------------------------------------------------------------------

#: An accepted name whose ``_of_ion_state`` locus claims a state resolution none
#: of its species-level sources has: the sources agree with each other and with
#: the DD, the NAME is the outlier.
_STATE_NAME = "atomic_count_of_ion_state"
_SPECIES_PATHS = (
    "core_profiles/profiles_1d/ion/element/atoms_n",
    "edge_profiles/profiles_1d/ion/element/atoms_n",
)


def _wipeout_rows(*, name_stage: str = "accepted") -> list[dict]:
    return [_row(p, _STATE_NAME, name_stage=name_stage) for p in _SPECIES_PATHS]


def test_uniform_wipeout_is_reported_as_a_name_defect() -> None:
    result = audit_attachments(_client(_wipeout_rows()))
    assert len(result.rejected) == 2
    assert len(result.names_misnamed) == 1
    d = result.names_misnamed[0]
    assert d.sn_id == _STATE_NAME
    assert d.rule == "state-resolution mismatch"
    assert d.attachment_count == 2
    assert d.example_dd_path in _SPECIES_PATHS
    assert d.name_stage == "accepted"


def test_uniform_wipeout_is_never_detached() -> None:
    """Renaming the name is the repair — detaching would rewind every source."""
    gc = _client(_wipeout_rows())
    result = reconcile_attachment_consistency(gc, include_accepted=True)
    assert len(result.rejected) == 2
    assert result.skipped_misnamed == 2
    assert result.detached == 0
    assert result.names_orphaned == []
    assert gc.query.call_count == 1, "only the read query may run"


def test_mixed_rule_wipeout_is_detached() -> None:
    """Attachments failing for DIFFERENT reasons are an incoherent source set.

    No single rename fixes them, so the ordinary detach-and-recompose path owns
    the repair and the post-detach orphan net reports the result.
    """
    gc = _client(
        [
            _row(_SPECIES_PATHS[0], _STATE_NAME),
            _row(
                "core_profiles/profiles_1d/ion/state/instant_changes/atoms_n",
                _STATE_NAME,
            ),
        ]
    )
    result = reconcile_attachment_consistency(gc)
    assert {v.rule for v in result.rejected} == {
        "state-resolution mismatch",
        "tense mismatch",
    }
    assert result.names_misnamed == []
    assert result.skipped_misnamed == 0
    assert result.detached == 2


def test_single_rejected_attachment_is_not_a_name_defect() -> None:
    """One source is no corroboration — and it is the case the detach was for."""
    gc = _client(
        [
            _row(
                "summary/boundary/strike_point_inner_z/value",
                "z_image_up_unit_vector_of_camera",
            )
        ]
    )
    result = reconcile_attachment_consistency(gc)
    assert result.names_misnamed == []
    assert result.detached == 1


def test_superseded_wipeout_is_not_reported_as_a_name_defect() -> None:
    """A deprecation stub's edges are provenance; there is nothing to rename."""
    gc = _client(_wipeout_rows(name_stage="superseded"))
    result = reconcile_attachment_consistency(gc, include_accepted=True)
    assert result.names_misnamed == []
    assert result.skipped_historical == 2
    assert result.detached == 0


def test_as_dict_reports_name_level_defects() -> None:
    result = audit_attachments(_client(_wipeout_rows()))
    d = result.as_dict()
    assert d["names_misnamed"] == 1
    assert d["misnamed"][0]["sn_id"] == _STATE_NAME
    assert d["misnamed"][0]["rule"] == "state-resolution mismatch"


# ---------------------------------------------------------------------------
# Live graph — detach, reroute, orphan reporting, idempotency
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc():
    from imas_codex.graph.client import GraphClient
    from imas_codex.settings import get_graph_uri

    uri = os.environ.get("IMAS_CODEX_TEST_NEO4J_URI")
    if not uri:
        pytest.skip("IMAS_CODEX_TEST_NEO4J_URI is not configured")
    if os.environ.get("IMAS_CODEX_TEST_NEO4J_EPHEMERAL") != "1":
        pytest.fail("attachment graph tests require an ephemeral graph")
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()
    if uri == project_uri:
        pytest.fail("attachment graph tests refuse the configured project graph")
    client = GraphClient(
        uri=uri,
        username="neo4j",
        password="",
        graph_name="ephemeral-attachment-audit",
    )
    try:
        client.get_stats()
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"Neo4j not available: {exc}")
    try:
        yield client
    finally:
        client.close()


class _GraphFixtureScope:
    """Track and remove only the exact graph entities created by one test."""

    _DELETE_ORDER = (
        "StandardNameChange",
        "StandardNameSourceRetry",
        "DocsRevision",
        "Review",
        "StandardNameSource",
        "IMASNode",
        "StandardName",
        "Unit",
    )

    def __init__(self, gc: Any) -> None:
        self.gc = gc
        self.prefix = f"{_FIXTURE_ID_STEM}{uuid.uuid4().hex}__"
        self.created: dict[str, set[str]] = {
            label: set() for label in self._DELETE_ORDER
        }
        self.cleaned = False

    def __enter__(self) -> _GraphFixtureScope:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        try:
            self.cleanup()
        except BaseException as cleanup_error:
            if exc is None:
                raise
            exc.add_note(f"attachment fixture cleanup also failed: {cleanup_error}")
        return False

    def path(self, suffix: str) -> str:
        return f"{self.prefix}{suffix}"

    def uid(self, tag: str) -> str:
        return self.path(f"{tag}_{uuid.uuid4().hex[:8]}")

    def track(self, label: str, *ids: str) -> None:
        self.created[label].update(identifier for identifier in ids if identifier)

    def _discover_changes(self) -> None:
        name_ids = sorted(self.created["StandardName"])
        dd_paths = sorted(self.created["IMASNode"])
        if not name_ids and not dd_paths:
            return
        rows = self.gc.query(
            """
            MATCH (change:StandardNameChange)
            WHERE change.from_name IN $dd_paths
               OR change.to_name IN $dd_paths
               OR EXISTS {
                    MATCH (sn:StandardName)-[:HAS_INTERNAL_CHANGE]->(change)
                    WHERE sn.id IN $name_ids
                  }
            RETURN DISTINCT change.id AS id
            """,
            name_ids=name_ids,
            dd_paths=dd_paths,
        )
        self.track("StandardNameChange", *(row["id"] for row in rows))
        source_ids = sorted(self.created["StandardNameSource"])
        if source_ids:
            retry_rows = self.gc.query(
                """
                MATCH (source:StandardNameSource)-[:HAS_RETRY_EVENT]->
                      (retry:StandardNameSourceRetry)
                WHERE source.id IN $source_ids
                RETURN DISTINCT retry.id AS id
                """,
                source_ids=source_ids,
            )
            self.track("StandardNameSourceRetry", *(row["id"] for row in retry_rows))

    def remaining_count(self) -> int:
        remaining = 0
        for label, ids in self.created.items():
            if not ids:
                continue
            rows = self.gc.query(
                f"MATCH (n:{label}) WHERE n.id IN $ids RETURN count(n) AS count",
                ids=sorted(ids),
            )
            remaining += int(rows[0]["count"] if rows else 0)
        return remaining

    def cleanup(self) -> None:
        if self.cleaned:
            assert self.remaining_count() == 0
            return
        self._discover_changes()
        all_ids = sorted(
            {identifier for ids in self.created.values() for identifier in ids}
        )
        if all_ids:
            self.gc.query(
                """
                MATCH (node)
                WHERE node.id IN $ids
                OPTIONAL MATCH (node)-[relationship]-()
                WITH collect(DISTINCT relationship) AS relationships
                FOREACH (relationship IN relationships | DELETE relationship)
                """,
                ids=all_ids,
            )
        for label in self._DELETE_ORDER:
            ids = sorted(self.created[label])
            if not ids:
                continue
            self.gc.query(
                f"MATCH (node:{label}) WHERE node.id IN $ids DELETE node",
                ids=ids,
            )
        assert self.remaining_count() == 0, (
            f"fixture cleanup left exact tracked nodes for {self.prefix}"
        )
        self.cleaned = True


@pytest.fixture()
def _clean(_gc):
    """Provide a unique exact-ID scope with exception-safe final cleanup."""
    with _GraphFixtureScope(_gc) as scope:
        yield scope


def _seed_attachment(
    scope: _GraphFixtureScope,
    *,
    dd_path: str,
    sn_id: str,
    name_stage: str = "drafted",
    dd_unit: str = "m",
    sn_unit: str = "m",
    source_status: str = "composed",
) -> str:
    """Create (source)-[:PRODUCED_NAME]->(name) with the DD-side projection."""
    source_node_id = f"dd:{dd_path}"
    scope.track("IMASNode", dd_path)
    scope.track("StandardName", sn_id)
    scope.track("StandardNameSource", source_node_id)
    scope.gc.query(
        """
        MERGE (dd:IMASNode {id: $dd_path})
          SET dd.unit = $dd_unit, dd.node_category = 'quantity'
        MERGE (sn:StandardName {id: $sn_id})
          SET sn.name_stage        = $name_stage,
              sn.docs_stage        = 'pending',
              sn.origin            = 'pipeline',
              sn.validation_status = 'valid',
              sn.unit              = $sn_unit,
              sn.description       = 'Seeded attachment fixture',
              // Append, so a name can be seeded with several sources.
              sn.source_paths      = CASE
                WHEN 'dd:' + $dd_path IN coalesce(sn.source_paths, [])
                THEN sn.source_paths
                ELSE coalesce(sn.source_paths, []) + ('dd:' + $dd_path) END
        MERGE (src:StandardNameSource {id: $source_node_id})
          SET src.status      = $source_status,
              src.source_type = 'dd',
              src.source_id   = $dd_path,
              src.produced_sn_id = $sn_id,
              src.attempt_count  = 3
        MERGE (src)-[:FROM_DD_PATH]->(dd)
        MERGE (src)-[:PRODUCED_NAME]->(sn)
        MERGE (dd)-[:HAS_STANDARD_NAME]->(sn)
        """,
        dd_path=dd_path,
        sn_id=sn_id,
        name_stage=name_stage,
        dd_unit=dd_unit,
        sn_unit=sn_unit,
        source_node_id=source_node_id,
        source_status=source_status,
    )
    return source_node_id


@pytest.mark.graph
def test_graph_scope_cleans_partial_setup_without_hiding_failure(_gc):
    """An interrupted multi-query setup preserves its error and leaves no nodes."""
    scope = None
    with pytest.raises(RuntimeError, match="injected setup failure") as captured:
        with _GraphFixtureScope(_gc) as active_scope:
            scope = active_scope
            dd_path = scope.path("partial_setup/path")
            sn_id = scope.uid("partial_setup_name")
            _seed_attachment(scope, dd_path=dd_path, sn_id=sn_id)
            unit_id = scope.path("partial_setup_unit")
            scope.track("Unit", unit_id)
            _gc.query(
                """
                MATCH (dd:IMASNode {id: $dd_path})
                MERGE (unit:Unit {id: $unit_id})
                MERGE (dd)-[:HAS_UNIT]->(unit)
                """,
                dd_path=dd_path,
                unit_id=unit_id,
            )
            raise RuntimeError("injected setup failure")

    assert str(captured.value) == "injected setup failure"
    assert scope is not None and scope.cleaned is True
    assert scope.remaining_count() == 0


def _scoped(gc, prefix: str, **params):
    """Audit only the fixture's own attachments (the graph is shared)."""
    from imas_codex.standard_names import attachment_audit as mod

    rows = gc.query(
        mod._ATTACHMENTS_QUERY.format(scope="WHERE sn.id STARTS WITH $prefix"),
        prefix=prefix,
        **params,
    )
    return list(rows)


class _ScopedClient:
    """Delegate to a real client, but narrow the audit read to the fixture set."""

    def __init__(self, gc, prefix: str) -> None:
        self._gc = gc
        self._prefix = prefix

    def query(self, q: str, **params):
        from imas_codex.standard_names import attachment_audit as mod

        if q == mod._ATTACHMENTS_QUERY.format(scope=""):
            return _scoped(self._gc, self._prefix, **params)
        return self._gc.query(q, **params)

    def close(self) -> None:  # pragma: no cover - never owned by the reconcile
        pass


@pytest.mark.graph
def test_reconcile_detaches_and_reroutes(_gc, _clean):
    """An inconsistent edge is detached and its source returns to compose."""
    dd_path = _clean.path("summary/boundary/strike_point_inner_z/value")
    sn_id = _clean.uid("z_image_up_unit_vector_of_camera")
    source_node_id = _seed_attachment(_clean, dd_path=dd_path, sn_id=sn_id)

    result = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))

    assert len(result.rejected) == 1
    assert result.rejected[0].rule == "locus/source device mismatch"
    assert result.detached == 1
    assert result.sources_rerouted == 1

    rows = _gc.query(
        """
        MATCH (src:StandardNameSource {id: $sid})
        OPTIONAL MATCH (src)-[pn:PRODUCED_NAME]->(:StandardName)
        OPTIONAL MATCH (:IMASNode {id: $dd})-[hsn:HAS_STANDARD_NAME]->(:StandardName)
        MATCH (sn:StandardName {id: $sn})
        RETURN src.status AS status, src.produced_sn_id AS mirror,
               src.attempt_count AS attempts, count(pn) AS produced,
               count(hsn) AS dd_edges, sn.source_paths AS paths
        """,
        sid=source_node_id,
        dd=dd_path,
        sn=sn_id,
    )
    r = rows[0]
    assert r["produced"] == 0, "PRODUCED_NAME edge must be gone"
    assert r["dd_edges"] == 0, "DD-side projection must be gone"
    assert r["mirror"] is None, "scalar mirror must be cleared with the edge"
    assert r["status"] == "extracted", "source must return to the generate pool"
    assert r["attempts"] == 0, "attempt budget must reset for the fresh compose"
    assert not r["paths"], "source_paths projection must drop the detached path"


@pytest.mark.graph
def test_source_backing_a_good_name_keeps_its_status(_gc, _clean):
    """Detaching a stale edge must not strand the correct name the source made.

    A source consolidation left pointing at two names — one the guard rejects,
    one it accepts — loses only the wrong edge; rewinding it to 'extracted'
    would orphan the good name.
    """
    bad_path = _clean.path("summary/boundary/strike_point_inner_z/value")
    bad_name = _clean.uid("z_image_up_unit_vector_of_camera")
    good_name = _clean.uid("vertical_coordinate_of_inner_strike_point")
    source_node_id = _seed_attachment(_clean, dd_path=bad_path, sn_id=bad_name)
    # Same source also produces a name the guard accepts.
    _clean.track("StandardName", good_name)
    _gc.query(
        """
        MERGE (sn:StandardName {id: $good})
          SET sn.name_stage = 'accepted', sn.docs_stage = 'pending',
              sn.origin = 'pipeline', sn.validation_status = 'valid',
              sn.unit = 'm', sn.description = 'Seeded good name'
        WITH sn MATCH (src:StandardNameSource {id: $sid})
        MERGE (src)-[:PRODUCED_NAME]->(sn)
        """,
        good=good_name,
        sid=source_node_id,
    )

    result = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))
    assert result.detached == 1
    assert result.sources_rerouted == 0

    rows = _gc.query(
        """
        MATCH (src:StandardNameSource {id: $sid})
        OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(sn:StandardName)
        RETURN src.status AS status, src.produced_sn_id AS mirror,
               src.attempt_count AS attempts, collect(sn.id) AS names
        """,
        sid=source_node_id,
    )
    r = rows[0]
    assert r["names"] == [good_name], "only the rejected edge may be removed"
    assert r["status"] == "composed", "a source with a live name is not rewound"
    assert r["attempts"] == 3, "attempt budget must survive when not rerouting"
    assert r["mirror"] == good_name, "the mirror must follow the surviving name"


@pytest.mark.graph
def test_superseded_provenance_survives(_gc, _clean):
    """Edges a supersede deliberately left in place are not touched."""
    dd_path = _clean.path("summary/boundary/strike_point_inner_z/value")
    sn_id = _clean.uid("z_image_up_unit_vector_of_camera")
    source_node_id = _seed_attachment(
        _clean, dd_path=dd_path, sn_id=sn_id, name_stage="superseded"
    )

    result = reconcile_attachment_consistency(
        _ScopedClient(_gc, _clean.prefix), include_accepted=True
    )
    assert result.skipped_historical == 1
    assert result.detached == 0

    rows = _gc.query(
        """
        MATCH (src:StandardNameSource {id: $sid})-[:PRODUCED_NAME]->(sn:StandardName)
        RETURN sn.id AS sn_id, src.status AS status
        """,
        sid=source_node_id,
    )
    assert rows[0]["sn_id"] == sn_id
    assert rows[0]["status"] == "composed"


@pytest.mark.graph
def test_reconcile_keeps_history(_gc, _clean):
    """Detaching records a StandardNameChange and deletes no node."""
    dd_path = _clean.path("summary/boundary/strike_point_inner_z/value")
    sn_id = _clean.uid("z_image_up_unit_vector_of_camera")
    _seed_attachment(_clean, dd_path=dd_path, sn_id=sn_id)

    reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))

    rows = _gc.query(
        """
        MATCH (sn:StandardName {id: $sn})-[:HAS_INTERNAL_CHANGE]->
              (c:StandardNameChange)
        WHERE c.operation = 'detach_inconsistent_attachment'
        RETURN c.from_name AS from_name, c.reason AS reason
        """,
        sn=sn_id,
    )
    assert len(rows) == 1
    assert rows[0]["from_name"] == dd_path
    assert "locus" in rows[0]["reason"].lower()

    survived = _gc.query(
        """
        MATCH (sn:StandardName {id: $sn})
        MATCH (src:StandardNameSource {id: $sid})
        MATCH (dd:IMASNode {id: $dd})
        RETURN count(*) AS c
        """,
        sn=sn_id,
        sid=f"dd:{dd_path}",
        dd=dd_path,
    )
    assert survived[0]["c"] == 1, "no node may be deleted by a detach"


@pytest.mark.graph
def test_reconcile_is_idempotent(_gc, _clean):
    """A second pass acts on nothing."""
    dd_path = _clean.path("summary/boundary/strike_point_inner_z/value")
    sn_id = _clean.uid("z_image_up_unit_vector_of_camera")
    _seed_attachment(_clean, dd_path=dd_path, sn_id=sn_id)

    first = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))
    assert first.detached == 1

    second = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))
    assert second.rejected == []
    assert second.detached == 0
    assert second.sources_rerouted == 0


@pytest.mark.graph
def test_reconcile_leaves_consistent_attachment_alone(_gc, _clean):
    dd_path = _clean.path("camera_ir/channel/camera/up/z")
    sn_id = _clean.uid("z_image_up_unit_vector_of_camera")
    source_node_id = _seed_attachment(
        _clean, dd_path=dd_path, sn_id=sn_id, dd_unit="m", sn_unit="1"
    )

    result = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))
    assert result.rejected == []
    assert result.detached == 0

    rows = _gc.query(
        """
        MATCH (src:StandardNameSource {id: $sid})-[:PRODUCED_NAME]->(sn:StandardName)
        RETURN src.status AS status, sn.id AS sn_id
        """,
        sid=source_node_id,
    )
    assert rows[0]["status"] == "composed"
    assert rows[0]["sn_id"] == sn_id


@pytest.mark.graph
def test_accepted_attachment_survives_without_the_flag(_gc, _clean):
    """Catalog-authoritative state is not broken casually."""
    dd_path = _clean.path("summary/boundary/strike_point_inner_z/value")
    sn_id = _clean.uid("z_image_up_unit_vector_of_camera")
    _seed_attachment(_clean, dd_path=dd_path, sn_id=sn_id, name_stage="accepted")

    guarded = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))
    assert guarded.skipped_protected == 1
    assert guarded.detached == 0

    forced = reconcile_attachment_consistency(
        _ScopedClient(_gc, _clean.prefix), include_accepted=True
    )
    assert forced.detached == 1
    assert forced.skipped_protected == 0


# ---------------------------------------------------------------------------
# Write-path coverage — every writer is either gated or reachable by the audit
# ---------------------------------------------------------------------------


def test_audit_covers_every_attachment_writer() -> None:
    """The audit's selector must match what every writer can produce.

    Only the compose paths in ``workers`` call the guard at write time. Every
    other writer — the refine successor migration, the edit rebind, the
    derived-parent seeders, the provenance rebuilders — MERGEs a
    ``PRODUCED_NAME`` edge with no gate, so the audit is their only coverage.
    It reaches an attachment through ``PRODUCED_NAME`` + ``FROM_DD_PATH``
    alone, which is the shape every one of those writers produces; this asserts
    the selector has not been narrowed to something a writer can slip past.
    """
    from imas_codex.standard_names import attachment_audit as mod

    q = mod._ATTACHMENTS_QUERY
    assert "MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)" in q
    assert "MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)" in q
    # No stage / origin / status narrowing on the selector: a laundered
    # attachment can sit on any stage, any origin, and any source status.
    for narrowing in (
        "name_stage IN",
        "name_stage =",
        "origin =",
        "src.status =",
        "{status:",
    ):
        assert narrowing not in q, (
            f"audit selector narrowed by {narrowing!r} — a writer that produces "
            "that state escapes retroactive validation"
        )


def test_attachment_read_aggregates_each_relationship_before_projection() -> None:
    """Unit and peer-name fan-out must not change attachment cardinality."""
    from imas_codex.standard_names import attachment_audit as mod

    q = mod._ATTACHMENTS_QUERY
    assert "collect(DISTINCT du.id) AS dd_relationship_units" in q
    assert "collect(DISTINCT nu.id) AS sn_relationship_units" in q
    assert "count(DISTINCT other) AS other_live_names" in q
    assert "coalesce(head(collect(DISTINCT nu.id)), sn.unit)" not in q
    assert q.index("AS dd_relationship_units") < q.index(
        "OPTIONAL MATCH (sn)-[:HAS_UNIT]"
    )
    assert q.index("AS sn_relationship_units") < q.index(
        "OPTIONAL MATCH (src)-[:PRODUCED_NAME]->(other:StandardName)"
    )


@pytest.mark.graph
def test_attachment_read_has_one_row_and_deterministic_unit_ambiguity(_gc, _clean):
    """Zero, one, or many unit edges each yield exactly one attachment row."""
    from imas_codex.standard_names import attachment_audit as mod

    scalar_unit = _clean.path("scalar_unit")
    edge_unit = _clean.path("edge_unit")
    conflicting_unit = _clean.path("conflicting_unit")
    cases = (
        ("no_edges", [], scalar_unit),
        ("one_edge", [edge_unit], edge_unit),
        ("many_edges", [edge_unit, conflicting_unit], None),
    )
    name_ids: list[str] = []
    for tag, units, _expected in cases:
        dd_path = _clean.path(f"unit_case/{tag}")
        sn_id = _clean.uid(tag)
        name_ids.append(sn_id)
        _seed_attachment(
            _clean,
            dd_path=dd_path,
            sn_id=sn_id,
            dd_unit=scalar_unit,
            sn_unit=scalar_unit,
        )
        for unit_id in units:
            _clean.track("Unit", unit_id)
            _gc.query(
                """
                MATCH (sn:StandardName {id: $sn_id})
                MATCH (dd:IMASNode {id: $dd_path})
                MERGE (unit:Unit {id: $unit_id})
                MERGE (sn)-[:HAS_UNIT]->(unit)
                MERGE (dd)-[:HAS_UNIT]->(unit)
                """,
                sn_id=sn_id,
                dd_path=dd_path,
                unit_id=unit_id,
            )

    scope = "WHERE sn.id IN $name_ids"
    params = {
        "name_ids": name_ids,
        "historical": sorted(mod._HISTORICAL_NAME_STAGES),
    }
    # EXPLAIN proves the same read is accepted by the configured Cypher parser.
    _gc.query(f"EXPLAIN {mod._ATTACHMENTS_QUERY.format(scope=scope)}", **params)
    rows = list(_gc.query(mod._ATTACHMENTS_QUERY.format(scope=scope), **params))

    assert len(rows) == len(cases)
    by_tag = {row["dd_path"].rsplit("/", 1)[-1]: row for row in rows}
    for tag, units, expected in cases:
        assert by_tag[tag]["dd_unit"] == expected
        assert sorted(by_tag[tag]["dd_relationship_units"]) == sorted(units)
        assert by_tag[tag]["sn_unit"] == expected
        assert sorted(by_tag[tag]["sn_relationship_units"]) == sorted(units)


def test_refine_successor_migration_is_gated() -> None:
    """The migration re-pairs a historical source set inside its transaction.

    ``persist_refined_name`` moves every ``PRODUCED_NAME`` and
    ``HAS_STANDARD_NAME`` edge from the predecessor onto a DIFFERENT name. The
    set is historical but the pairing is new, and a new pairing is what the
    guard judges — ungated, this channel launders an edge that compose would
    have refused. Every source must be admitted before the first lineage or
    mirror mutation, on the same query handle that owns the transaction.
    """
    import inspect

    from imas_codex.standard_names import graph_ops

    src = inspect.getsource(graph_ops.persist_refined_name)
    assert "REFINE_ATOMIC_PREFLIGHT" in src
    assert "guard_source_pairings(" in src
    assert "query_handle, new_name, candidate_source_ids" in src
    assert "guarded.rejected" in src
    assert "retarget_standard_name_sources(" in src
    assert "source_ids=candidate_source_ids" in src
    assert "gate_migrated_attachments(sn_id=new_name)" not in src


def test_gate_validates_only_the_successor() -> None:
    """The gate reads one name, not the corpus — it runs on every refine."""
    from imas_codex.standard_names.attachment_audit import gate_migrated_attachments

    other = "electron_density"
    gc = _client(
        [
            _row(
                "summary/boundary/strike_point_inner_z/value",
                "z_image_up_unit_vector_of_camera",
            ),
            _row(
                "core_profiles/profiles_1d/electrons/density",
                other,
                dd_unit="m^-3",
                sn_unit="m^-3",
            ),
        ]
    )
    result = gate_migrated_attachments(gc, sn_id=other)
    assert result.checked == 1, "the gate must not audit names it did not touch"
    assert result.rejected == []


def test_gate_detaches_a_laundered_migration() -> None:
    """A migrated edge the guard refuses is detached, not left on the successor."""
    from imas_codex.standard_names.attachment_audit import gate_migrated_attachments

    sn_id = "z_image_up_unit_vector_of_camera"
    gc = _client([_row("summary/boundary/strike_point_inner_z/value", sn_id)])
    result = gate_migrated_attachments(gc, sn_id=sn_id)
    assert [v.rule for v in result.rejected] == ["locus/source device mismatch"]
    assert result.detached == 1


# ---------------------------------------------------------------------------
# Targeted detach — the semantic mis-share the guard cannot judge
# ---------------------------------------------------------------------------


def _terminal_recovery_client(*, eligible: bool = True) -> tuple[MagicMock, MagicMock]:
    gc = MagicMock()
    tx = MagicMock()
    tx.closed = False
    tx.run.return_value = (
        [
            {
                "source_node_id": "dd:spectrometer/channel/isotope_ratio",
                "sn_id": "hydrogen_fraction",
                "name_stage": "superseded",
                "retry_event_id": "source-retry:test",
                "change_event_id": "sn-change:test",
            }
        ]
        if eligible
        else []
    )
    session = MagicMock()
    session.begin_transaction.return_value = tx
    gc.session.return_value.__enter__.return_value = session
    gc.query.return_value = (
        [
            {
                "source_node_id": "dd:spectrometer/channel/isotope_ratio",
                "source_status": "composed",
                "attempt_count": 3,
                "last_error": None,
                "name_stage": "superseded",
            }
        ]
        if eligible
        else []
    )
    return gc, tx


def test_terminal_recovery_is_one_exact_transaction() -> None:
    from imas_codex.standard_names.attachment_audit import (
        recover_terminal_attachment,
    )

    path = "spectrometer/channel/isotope_ratio"
    gc, tx = _terminal_recovery_client()
    result = recover_terminal_attachment(
        path,
        "hydrogen_fraction",
        reason="the candidate resolved to historical lineage",
        gc=gc,
    )

    assert result == {
        "ok": True,
        "dd_path": path,
        "sn_id": "hydrogen_fraction",
        "source_node_id": f"dd:{path}",
        "dry_run": False,
        "name_stage": "superseded",
        "retry_event_id": "source-retry:test",
        "change_event_id": "sn-change:test",
    }
    assert tx.run.call_count == 1
    cypher = tx.run.call_args.args[0]
    assert "src.status IN $source_statuses" in cypher
    assert "src.claim_token IS NULL" in cypher
    assert "src.produced_sn_id = sn.id" in cypher
    assert "COUNT { (src)-[:PRODUCED_NAME]->(:StandardName) } = 1" in cypher
    assert "MATCH (dd)-[hsn:HAS_STANDARD_NAME]->(sn)" in cypher
    assert "DELETE pn, hsn" in cypher
    assert "src.status = 'extracted'" in cypher
    assert "src.attempt_count = 0" in cypher
    assert "CREATE (retry:StandardNameSourceRetry" in cypher
    assert "MERGE (src)-[:HAS_RETRY_EVENT]->(retry)" in cypher
    assert "CREATE (change:StandardNameChange" in cypher
    assert "MERGE (sn)-[:HAS_INTERNAL_CHANGE]->(change)" in cypher
    assert "terminal_stage" in cypher
    params = tx.run.call_args.kwargs
    assert params["source_node_id"] == f"dd:{path}"
    assert params["dd_path"] == path
    assert params["sn_id"] == "hydrogen_fraction"
    assert params["source_statuses"] == ["attached", "composed"]
    tx.commit.assert_called_once()


def test_terminal_recovery_dry_run_has_no_write_transaction() -> None:
    from imas_codex.standard_names.attachment_audit import (
        recover_terminal_attachment,
    )

    gc, tx = _terminal_recovery_client()
    result = recover_terminal_attachment(
        "spectrometer/channel/isotope_ratio",
        "hydrogen_fraction",
        reason="inspect exact historical target",
        gc=gc,
        dry_run=True,
    )

    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["name_stage"] == "superseded"
    assert "DELETE" not in gc.query.call_args.args[0]
    gc.session.assert_not_called()
    tx.run.assert_not_called()


def test_terminal_recovery_fails_closed_without_partial_events() -> None:
    from imas_codex.standard_names.attachment_audit import (
        recover_terminal_attachment,
    )

    gc, tx = _terminal_recovery_client(eligible=False)
    result = recover_terminal_attachment(
        "spectrometer/channel/isotope_ratio",
        "hydrogen_fraction",
        reason="target must remain terminal and exact",
        gc=gc,
    )

    assert result["ok"] is False
    assert "changed or is ineligible" in result["reason"]
    assert tx.run.call_count == 1
    tx.commit.assert_called_once()
    assert tx.close.call_count == 0


def test_terminal_recovery_exception_rolls_back_edges_and_events() -> None:
    from imas_codex.standard_names.attachment_audit import (
        recover_terminal_attachment,
    )

    gc, tx = _terminal_recovery_client()
    tx.run.side_effect = RuntimeError("transaction failed")
    with pytest.raises(RuntimeError, match="transaction failed"):
        recover_terminal_attachment(
            "spectrometer/channel/isotope_ratio",
            "hydrogen_fraction",
            reason="recover exact historical target",
            gc=gc,
        )

    tx.commit.assert_not_called()
    tx.close.assert_called_once()


def test_detach_terminal_recovery_routes_to_exact_transaction() -> None:
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc, tx = _terminal_recovery_client()
    result = detach_one_attachment(
        "spectrometer/channel/isotope_ratio",
        "hydrogen_fraction",
        reason="historical collision",
        gc=gc,
        terminal_recovery=True,
    )

    assert result["ok"] is True
    assert result["retry_event_id"] == "source-retry:test"
    tx.commit.assert_called_once()


def _terminal_graph_state(
    *,
    source_paths: list[str] | None = None,
    extra_target_sources: int = 0,
    extra_target_projections: int = 0,
    from_dd_path: bool = True,
    produced_edge: bool = True,
    projected_edge: bool = True,
    scalar_matches: bool = True,
    source_status: str = "composed",
    claimed: bool = False,
    name_stage: str = "superseded",
) -> dict:
    """Small graph model for exercising the recovery predicate and mutation."""
    path = "spectrometer/channel/isotope_ratio"
    source_id = f"dd:{path}"
    target_id = "hydrogen_fraction"
    target_sources = [source_id] if produced_edge else []
    target_sources.extend(f"dd:other/source/{i}" for i in range(extra_target_sources))
    target_projections = [path] if projected_edge else []
    target_projections.extend(
        f"other/projection/{i}" for i in range(extra_target_projections)
    )
    return {
        "path": path,
        "source_id": source_id,
        "target_id": target_id,
        "from_dd_path": from_dd_path,
        "source_outputs": [target_id] if produced_edge else [],
        "target_sources": target_sources,
        "target_projections": target_projections,
        "source": {
            "source_type": "dd",
            "source_id": path,
            "status": source_status,
            "produced_sn_id": target_id if scalar_matches else "another_name",
            "attempt_count": 7,
            "last_error": "candidate collided with terminal lineage",
            "composed_at": "2026-07-31T12:00:00Z",
            "claimed_at": "2026-07-31T11:59:00Z" if claimed else None,
            "claim_token": "claim:test" if claimed else None,
            "failed_at": "2026-07-31T11:58:00Z",
            "retry_events": ["source-retry:earlier"],
        },
        "target": {
            "name_stage": name_stage,
            "source_paths": None if source_paths is None else list(source_paths),
            "unit": "1",
            "reviewer_score_name": 0.73125,
            "grammar_version": "0.8.0rc63",
            "physics_domain": "spectroscopy",
            "cocos": 17,
            "refined_from": "hydrogen_fraction_legacy",
            "vocab_gap_ids": ["gap:preserved"],
        },
        "retries": [],
        "changes": [],
        "source_retry_links": [],
        "name_change_links": [],
    }


def _state_is_terminal_recovery_eligible(state: dict, params: dict) -> bool:
    """Evaluate the Cypher predicate against the small graph model."""
    source = state["source"]
    target = state["target"]
    path = params["dd_path"]
    target_id = params["sn_id"]
    source_id = params["source_node_id"]
    paths = target["source_paths"] or []
    mirror_exact = path in paths or f"dd:{path}" in paths
    mirror_missing_but_isolated = (
        not paths
        and len(state["target_sources"]) == 1
        and len(state["target_projections"]) == 1
    )
    return all(
        (
            source_id == state["source_id"],
            target_id == state["target_id"],
            state["from_dd_path"],
            source["source_type"] == "dd",
            source["source_id"] == path,
            source["status"] in params["source_statuses"],
            source["claimed_at"] is None,
            source["claim_token"] is None,
            source["produced_sn_id"] == target_id,
            target["name_stage"] in params["terminal_stages"],
            state["source_outputs"] == [target_id],
            source_id in state["target_sources"],
            path in state["target_projections"],
            mirror_exact or mirror_missing_but_isolated,
        )
    )


class _StatefulTerminalTransaction:
    """Transaction model that exposes partial-write and commit rollback defects."""

    def __init__(
        self,
        client: _StatefulTerminalClient,
        *,
        fail_statement: bool,
        fail_commit: bool,
    ) -> None:
        self.client = client
        self.fail_statement = fail_statement
        self.fail_commit = fail_commit
        self.closed = False
        self.committed = False
        self.rolled_back = False
        self.run_count = 0
        self.working = deepcopy(client.state)

    def run(self, _query: str, **params):
        self.run_count += 1
        if self.fail_statement:
            raise RuntimeError("transaction statement failed")
        if not _state_is_terminal_recovery_eligible(self.working, params):
            return []

        source = self.working["source"]
        target = self.working["target"]
        previous_status = source["status"]
        previous_attempt_count = source["attempt_count"]
        previous_error = source["last_error"]
        terminal_stage = target["name_stage"]
        target_id = self.working["target_id"]
        path = self.working["path"]
        source_id = self.working["source_id"]

        self.working["source_outputs"].remove(target_id)
        self.working["target_sources"].remove(source_id)
        self.working["target_projections"].remove(path)
        target["source_paths"] = [
            item
            for item in target["source_paths"] or []
            if item not in (path, f"dd:{path}")
        ]

        event_reason = (
            f'{params["reason"]} [terminal target "{target_id}" '
            f'at name_stage "{terminal_stage}"]'
        )
        retry = {
            "id": params["retry_event_id"],
            "source_id": source_id,
            "previous_status": previous_status,
            "previous_attempt_count": previous_attempt_count,
            "previous_error": previous_error,
            "reason": event_reason,
        }
        change = {
            "id": params["change_event_id"],
            "from_name": target_id,
            "operation": "recover_terminal_source_binding",
            "reason": event_reason,
            "origin": "terminal_binding_recovery",
            "internal": True,
        }
        self.working["retries"].append(retry)
        self.working["changes"].append(change)
        self.working["source_retry_links"].append(retry["id"])
        self.working["name_change_links"].append(change["id"])
        source["retry_events"].append(retry["id"])
        source.update(
            {
                "status": "extracted",
                "produced_sn_id": None,
                "composed_at": None,
                "attempt_count": 0,
                "claimed_at": None,
                "claim_token": None,
                "failed_at": None,
                "last_error": None,
            }
        )
        return [
            {
                "source_node_id": source_id,
                "sn_id": target_id,
                "name_stage": terminal_stage,
                "retry_event_id": retry["id"],
                "change_event_id": change["id"],
            }
        ]

    def commit(self) -> None:
        if self.fail_commit:
            raise RuntimeError("transaction commit failed")
        self.client.state = self.working
        self.committed = True
        self.closed = True

    def close(self) -> None:
        self.closed = True
        self.rolled_back = not self.committed


class _StatefulTerminalSession:
    def __init__(self, client: _StatefulTerminalClient) -> None:
        self.client = client

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def begin_transaction(self) -> _StatefulTerminalTransaction:
        tx = _StatefulTerminalTransaction(
            self.client,
            fail_statement=self.client.fail_statement,
            fail_commit=self.client.fail_commit,
        )
        self.client.last_tx = tx
        return tx


class _StatefulTerminalClient:
    def __init__(
        self,
        state: dict,
        *,
        fail_statement: bool = False,
        fail_commit: bool = False,
    ) -> None:
        self.state = deepcopy(state)
        self.fail_statement = fail_statement
        self.fail_commit = fail_commit
        self.query_count = 0
        self.session_count = 0
        self.last_tx: _StatefulTerminalTransaction | None = None

    def query(self, _query: str, **params):
        self.query_count += 1
        if not _state_is_terminal_recovery_eligible(self.state, params):
            return []
        source = self.state["source"]
        return [
            {
                "source_node_id": self.state["source_id"],
                "source_status": source["status"],
                "attempt_count": source["attempt_count"],
                "last_error": source["last_error"],
                "name_stage": self.state["target"]["name_stage"],
            }
        ]

    def session(self) -> _StatefulTerminalSession:
        self.session_count += 1
        return _StatefulTerminalSession(self)


def _recover_with_state(client: _StatefulTerminalClient, *, dry_run: bool = False):
    from imas_codex.standard_names.attachment_audit import recover_terminal_attachment

    return recover_terminal_attachment(
        client.state["path"],
        client.state["target_id"],
        reason="retry the exact source after a terminal collision",
        gc=client,
        dry_run=dry_run,
    )


@pytest.mark.parametrize("source_paths", [None, []])
def test_terminal_recovery_empty_mirror_dry_run_is_read_only(
    source_paths: list[str] | None,
) -> None:
    client = _StatefulTerminalClient(_terminal_graph_state(source_paths=source_paths))
    before = deepcopy(client.state)

    result = _recover_with_state(client, dry_run=True)

    assert result["ok"] is True
    assert result["previous_attempt_count"] == 7
    assert client.state == before
    assert client.query_count == 1
    assert client.session_count == 0


def test_terminal_recovery_empty_mirror_resets_exact_isolated_binding() -> None:
    client = _StatefulTerminalClient(_terminal_graph_state(source_paths=[]))
    protected = {
        key: deepcopy(value)
        for key, value in client.state["target"].items()
        if key != "source_paths"
    }

    result = _recover_with_state(client)

    assert result["ok"] is True
    assert client.state["source_outputs"] == []
    assert client.state["target_sources"] == []
    assert client.state["target_projections"] == []
    assert client.state["target"]["source_paths"] == []
    assert client.state["source"]["status"] == "extracted"
    assert client.state["source"]["produced_sn_id"] is None
    assert client.state["source"]["attempt_count"] == 0
    assert client.state["source"]["claimed_at"] is None
    assert len(client.state["retries"]) == 1
    assert len(client.state["changes"]) == 1
    assert {
        key: value
        for key, value in client.state["target"].items()
        if key != "source_paths"
    } == protected


def test_terminal_recovery_exact_mirror_preserves_unrelated_entries() -> None:
    path = "spectrometer/channel/isotope_ratio"
    client = _StatefulTerminalClient(
        _terminal_graph_state(source_paths=[f"dd:{path}", "dd:unrelated/path", path])
    )

    assert _recover_with_state(client)["ok"] is True
    assert client.state["target"]["source_paths"] == ["dd:unrelated/path"]


def test_terminal_recovery_accepts_attached_source_status() -> None:
    client = _StatefulTerminalClient(
        _terminal_graph_state(source_paths=[], source_status="attached")
    )

    result = _recover_with_state(client)

    assert result["ok"] is True
    assert client.state["source"]["status"] == "extracted"
    assert client.state["retries"][0]["previous_status"] == "attached"


@pytest.mark.parametrize(
    "source_path_entry",
    ["spectrometer/channel/isotope_ratio", "dd:spectrometer/channel/isotope_ratio"],
)
def test_terminal_recovery_accepts_each_exact_mirror_form(
    source_path_entry: str,
) -> None:
    client = _StatefulTerminalClient(
        _terminal_graph_state(source_paths=[source_path_entry])
    )

    assert _recover_with_state(client, dry_run=True)["ok"] is True


def test_terminal_recovery_queries_share_the_isolated_empty_mirror_guard() -> None:
    from imas_codex.standard_names import attachment_audit as mod

    guard = """OR (
      size(coalesce(sn.source_paths, [])) = 0
      AND COUNT { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) } = 1
      AND COUNT { (:IMASNode)-[:HAS_STANDARD_NAME]->(sn) } = 1
    )"""
    assert mod._TERMINAL_RECOVERY_ELIGIBILITY_QUERY.count(guard) == 1
    assert mod._TERMINAL_RECOVERY_QUERY.count(guard) == 1


def test_terminal_recovery_nonempty_unrelated_mirror_is_blocked() -> None:
    client = _StatefulTerminalClient(
        _terminal_graph_state(source_paths=["dd:unrelated/path"])
    )
    before = deepcopy(client.state)

    result = _recover_with_state(client, dry_run=True)

    assert result["ok"] is False
    assert client.state == before
    assert client.session_count == 0


@pytest.mark.parametrize(
    ("state_changes", "description"),
    [
        ({"extra_target_sources": 1}, "multiple target sources"),
        ({"extra_target_projections": 1}, "multiple target projections"),
    ],
)
def test_terminal_recovery_empty_mirror_requires_isolated_target(
    state_changes: dict, description: str
) -> None:
    client = _StatefulTerminalClient(
        _terminal_graph_state(source_paths=[], **state_changes)
    )
    before = deepcopy(client.state)

    result = _recover_with_state(client, dry_run=True)

    assert result["ok"] is False, description
    assert client.state == before


@pytest.mark.parametrize(
    "state_changes",
    [
        {"from_dd_path": False},
        {"produced_edge": False},
        {"projected_edge": False},
        {"scalar_matches": False},
        {"name_stage": "accepted"},
        {"source_status": "extracted"},
        {"claimed": True},
    ],
)
def test_terminal_recovery_still_requires_every_authoritative_predicate(
    state_changes: dict,
) -> None:
    client = _StatefulTerminalClient(
        _terminal_graph_state(source_paths=[], **state_changes)
    )
    before = deepcopy(client.state)

    result = _recover_with_state(client, dry_run=True)

    assert result["ok"] is False
    assert client.state == before


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("statement", "transaction statement failed"),
        ("commit", "transaction commit failed"),
    ],
)
def test_terminal_recovery_transaction_failures_roll_back_all_state(
    failure: str, message: str
) -> None:
    client = _StatefulTerminalClient(
        _terminal_graph_state(source_paths=[]),
        fail_statement=failure == "statement",
        fail_commit=failure == "commit",
    )
    before = deepcopy(client.state)

    with pytest.raises(RuntimeError, match=message):
        _recover_with_state(client)

    assert client.state == before
    assert client.last_tx is not None
    assert client.last_tx.rolled_back is True
    assert client.last_tx.committed is False


def test_terminal_recovery_ledgers_capture_previous_source_state_and_links() -> None:
    client = _StatefulTerminalClient(_terminal_graph_state(source_paths=[]))

    result = _recover_with_state(client)

    retry = client.state["retries"][0]
    change = client.state["changes"][0]
    assert retry == {
        "id": result["retry_event_id"],
        "source_id": client.state["source_id"],
        "previous_status": "composed",
        "previous_attempt_count": 7,
        "previous_error": "candidate collided with terminal lineage",
        "reason": (
            "retry the exact source after a terminal collision [terminal target "
            '"hydrogen_fraction" at name_stage "superseded"]'
        ),
    }
    assert change == {
        "id": result["change_event_id"],
        "from_name": "hydrogen_fraction",
        "operation": "recover_terminal_source_binding",
        "reason": retry["reason"],
        "origin": "terminal_binding_recovery",
        "internal": True,
    }
    assert client.state["source_retry_links"] == [retry["id"]]
    assert client.state["name_change_links"] == [change["id"]]
    assert client.state["source"]["retry_events"] == [
        "source-retry:earlier",
        retry["id"],
    ]


def test_terminal_recovery_repeat_refuses_without_duplicate_events() -> None:
    client = _StatefulTerminalClient(_terminal_graph_state(source_paths=[]))

    first = _recover_with_state(client)
    second = _recover_with_state(client)

    assert first["ok"] is True
    assert second["ok"] is False
    assert len(client.state["retries"]) == 1
    assert len(client.state["changes"]) == 1
    assert client.state["source_retry_links"] == [first["retry_event_id"]]
    assert client.state["name_change_links"] == [first["change_event_id"]]


# ---------------------------------------------------------------------------
# Exact batch recovery — manifest, CAS, rollback, and idempotence
# ---------------------------------------------------------------------------


_TERMINAL_RECOVERY_FIXTURES = (
    (
        "core_instant_changes/change/profiles_1d/rotation_frequency_tor_sonic",
        "toroidal_change_in_rotation_frequency_due_to_e_cross_b_drift",
        "composed",
    ),
    (
        "gas_injection/pipe/species/fraction",
        "ratio_of_neutral_species_gas_count_to_total_gas_count",
        "attached",
    ),
    (
        "gas_injection/valve/species/fraction",
        "ratio_of_neutral_species_gas_count_to_total_gas_count",
        "composed",
    ),
    (
        "mhd_linear/time_slice/toroidal_mode/plasma/psi_potential_perturbed",
        "perturbed_magnetic_vector_potential",
        "composed",
    ),
    (
        "pellets/time_slice/pellet/species/fraction",
        "ratio_of_neutral_species_gas_count_to_total_gas_count",
        "attached",
    ),
    (
        "pulse_schedule/density_control/valve/species/fraction",
        "ratio_of_neutral_species_gas_count_to_total_gas_count",
        "attached",
    ),
    (
        "pulse_schedule/nbi/unit/species/fraction",
        "ratio_of_neutral_species_gas_count_to_total_gas_count",
        "attached",
    ),
    (
        "spi/injector/fragmentation_gas/species/fraction",
        "ratio_of_neutral_species_gas_count_to_total_gas_count",
        "attached",
    ),
    (
        "spi/injector/propellant_gas/species/fraction",
        "ratio_of_neutral_species_gas_count_to_total_gas_count",
        "attached",
    ),
)


def _recovery_relationship(
    element_id: str,
    relationship_type: str,
    direction: str,
    other_element_id: str,
    other_labels: list[str],
    other_id: str,
    other_properties: dict[str, Any],
) -> dict[str, Any]:
    return {
        "element_id": element_id,
        "type": relationship_type,
        "direction": direction,
        "properties": {},
        "other_element_id": other_element_id,
        "other_labels": other_labels,
        "other_id": other_id,
        "other_properties": deepcopy(other_properties),
    }


def _terminal_recovery_closure_row(
    dd_path: str, sn_id: str, status: str
) -> dict[str, Any]:
    source_id = f"dd:{dd_path}"
    safe_id = source_id.replace("/", "_").replace(":", "_")
    source_element_id = f"source:{safe_id}"
    node_element_id = f"node:{dd_path}"
    name_element_id = f"name:{sn_id}"
    source_properties = {
        "id": source_id,
        "source_type": "dd",
        "source_id": dd_path,
        "status": status,
        "produced_sn_id": sn_id,
        "attempt_count": 4,
        "last_error": "terminal collision",
        "claimed_at": None,
        "claim_token": None,
        "dd_version": "4.1.1",
        "dd_snapshot_pinned": True,
        "dd_unit": "1",
        "physics_domain": "general",
        "retry_events": [],
    }
    node_properties = {
        "id": dd_path,
        "node_category": "quantity",
        "unit": "1",
        "cocos": 17,
    }
    name_properties = {
        "id": sn_id,
        "name_stage": "superseded",
        "docs_stage": "accepted",
        "status": "draft",
        "validation_status": "valid",
        "reviewer_score_name": 0.93,
        "reviewer_score_docs": 0.95,
        "unit": "1",
        "cocos": 17,
        "origin": "pipeline",
        "source_paths": [source_id, "dd:unrelated/preserved"],
    }
    source_relationships = [
        _recovery_relationship(
            f"from:{safe_id}",
            "FROM_DD_PATH",
            "out",
            node_element_id,
            ["IMASNode"],
            dd_path,
            node_properties,
        ),
        _recovery_relationship(
            f"binding:{safe_id}",
            "PRODUCED_NAME",
            "out",
            name_element_id,
            ["StandardName"],
            sn_id,
            name_properties,
        ),
    ]
    node_relationships = [
        _recovery_relationship(
            f"from:{safe_id}",
            "FROM_DD_PATH",
            "in",
            source_element_id,
            ["StandardNameSource"],
            source_id,
            source_properties,
        ),
        _recovery_relationship(
            f"projection:{safe_id}",
            "HAS_STANDARD_NAME",
            "out",
            name_element_id,
            ["StandardName"],
            sn_id,
            name_properties,
        ),
        _recovery_relationship(
            f"dd-unit:{safe_id}",
            "HAS_UNIT",
            "out",
            "unit:1",
            ["Unit"],
            "1",
            {"id": "1", "symbol": "1"},
        ),
    ]
    name_relationships = [
        _recovery_relationship(
            f"binding:{safe_id}",
            "PRODUCED_NAME",
            "in",
            source_element_id,
            ["StandardNameSource"],
            source_id,
            source_properties,
        ),
        _recovery_relationship(
            f"projection:{safe_id}",
            "HAS_STANDARD_NAME",
            "in",
            node_element_id,
            ["IMASNode"],
            dd_path,
            node_properties,
        ),
        _recovery_relationship(
            f"review:{safe_id}",
            "HAS_REVIEW",
            "out",
            f"review-node:{safe_id}",
            ["Review"],
            f"review:{source_id}",
            {"id": f"review:{source_id}", "score": 0.93},
        ),
        _recovery_relationship(
            f"lineage:{safe_id}",
            "REFINED_FROM",
            "out",
            f"lineage-name:{safe_id}",
            ["StandardName"],
            f"{sn_id}_predecessor",
            {"id": f"{sn_id}_predecessor", "name_stage": "superseded"},
        ),
    ]
    return {
        "source_id": source_id,
        "dd_path": dd_path,
        "sn_id": sn_id,
        "sources": [
            {
                "element_id": source_element_id,
                "labels": ["StandardNameSource"],
                "properties": source_properties,
                "relationships": source_relationships,
            }
        ],
        "nodes": [
            {
                "element_id": node_element_id,
                "labels": ["IMASNode"],
                "properties": node_properties,
                "relationships": node_relationships,
            }
        ],
        "names": [
            {
                "element_id": name_element_id,
                "labels": ["StandardName"],
                "properties": name_properties,
                "relationships": name_relationships,
            }
        ],
    }


def _terminal_recovery_manifest_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from imas_codex.standard_names import attachment_audit as mod

    cohort_context: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        target = cohort_context.setdefault(
            row["sn_id"],
            {
                "source_ids": set(),
                "dd_paths": set(),
                "retry_event_ids": set(),
                "change_event_ids": set(),
            },
        )
        target["source_ids"].add(row["source_id"])
        target["dd_paths"].add(row["dd_path"])
        target["retry_event_ids"].add(f"future-retry:{row['source_id']}")
        target["change_event_ids"].add(f"future-change:{row['source_id']}")
    manifest_rows = []
    for row in rows:
        hashes = mod._terminal_recovery_snapshot_hashes(
            row,
            retry_event_id=f"future-retry:{row['source_id']}",
            change_event_id=f"future-change:{row['source_id']}",
            cohort_context=cohort_context,
        )
        manifest_rows.append(
            {
                "operation": "recover_terminal_attachment",
                "source_id": row["source_id"],
                "dd_path": row["dd_path"],
                "sn_id": row["sn_id"],
                "expected_source_status": row["sources"][0]["properties"]["status"],
                "expected_name_stage": row["names"][0]["properties"]["name_stage"],
                "expected_closure_hash": hashes["closure_hash"],
                "expected_preserved_state_hash": hashes["preserved_state_hash"],
                "expected_participant_ids_hash": hashes["participant_ids_hash"],
                "expected_relationship_ids_hash": hashes["relationship_ids_hash"],
                "west_intersection": 0,
                "test_intersection": 0,
            }
        )
    return {
        "schema": "imas-codex.terminal-attachment-recovery-manifest",
        "schema_version": 1,
        "operation": "recover_terminal_attachment",
        "rows": manifest_rows,
    }


def _share_terminal_recovery_targets(rows: list[dict[str, Any]]) -> None:
    """Make repeated target ids one shared in-memory graph participant."""
    by_target: dict[str, dict[str, Any]] = {}
    for row in rows:
        target_id = row["sn_id"]
        candidate = row["names"][0]
        shared = by_target.get(target_id)
        if shared is None:
            by_target[target_id] = candidate
            continue
        existing_relationship_ids = {
            relationship["element_id"] for relationship in shared["relationships"]
        }
        shared["relationships"].extend(
            relationship
            for relationship in candidate["relationships"]
            if relationship["type"] in {"PRODUCED_NAME", "HAS_STANDARD_NAME"}
            and relationship["element_id"] not in existing_relationship_ids
        )
        for source_path in candidate["properties"]["source_paths"]:
            if source_path not in shared["properties"]["source_paths"]:
                shared["properties"]["source_paths"].append(source_path)
        row["names"] = [shared]


def _write_terminal_recovery_manifest(path: Path, rows: list[dict[str, Any]]) -> str:
    _share_terminal_recovery_targets(rows)
    path.write_text(
        json.dumps(_terminal_recovery_manifest_payload(rows), sort_keys=True),
        encoding="utf-8",
    )
    return sha256(path.read_bytes()).hexdigest()


class _BatchRecoveryTransaction:
    def __init__(
        self,
        client: _BatchRecoveryClient,
        *,
        race: bool,
        partial: bool,
    ) -> None:
        self.client = client
        self.working = deepcopy(client.rows)
        self.race = race
        self.partial = partial
        self.raced = False
        self.committed = False
        self.rolled_back = False

    def run(self, query: str, **params: Any):
        if "TERMINAL_ATTACHMENT_RECOVERY_SOURCES" in query:
            return [
                {
                    "source_id": row["source_id"],
                    "participants": deepcopy(row["sources"]),
                }
                for row in sorted(self.working, key=lambda item: item["source_id"])
            ]
        if "TERMINAL_ATTACHMENT_RECOVERY_NODES" in query:
            return [
                {
                    "participant_id": row["dd_path"],
                    "participants": deepcopy(row["nodes"]),
                }
                for row in sorted(self.working, key=lambda item: item["dd_path"])
            ]
        if "TERMINAL_ATTACHMENT_RECOVERY_NAMES" in query:
            by_name: dict[str, list[dict[str, Any]]] = {}
            for row in self.working:
                by_name.setdefault(row["sn_id"], row["names"])
            return [
                {"participant_id": sn_id, "participants": deepcopy(by_name[sn_id])}
                for sn_id in sorted(by_name)
            ]
        if "SOURCE_SNAPSHOT_MIGRATION_LOCK" in query:
            if self.race and not self.raced:
                self.working[0]["names"][0]["relationships"][-1]["properties"][
                    "concurrent"
                ] = True
                self.raced = True
            return [{"locked": len(params["element_ids"])}]
        if "TERMINAL_ATTACHMENT_RECOVERY_RELATIONSHIP_LOCK" in query:
            return [{"locked": len(params["relationships"])}]
        if "TERMINAL_ATTACHMENT_RECOVERY_APPLY" not in query:
            raise AssertionError(query)
        source_ids: list[str] = []
        retry_ids: list[str] = []
        change_ids: list[str] = []
        for item in params["items"]:
            row = next(
                candidate
                for candidate in self.working
                if candidate["source_id"] == item["source_id"]
            )
            source = row["sources"][0]
            node = row["nodes"][0]
            name = row["names"][0]
            previous = deepcopy(source["properties"])
            source["relationships"] = [
                relationship
                for relationship in source["relationships"]
                if not (
                    relationship["type"] == "PRODUCED_NAME"
                    and relationship["other_id"] == item["sn_id"]
                )
            ]
            node["relationships"] = [
                relationship
                for relationship in node["relationships"]
                if not (
                    relationship["type"] == "HAS_STANDARD_NAME"
                    and relationship["other_id"] == item["sn_id"]
                )
            ]
            name["relationships"] = [
                relationship
                for relationship in name["relationships"]
                if not (
                    relationship["type"] in {"PRODUCED_NAME", "HAS_STANDARD_NAME"}
                    and relationship["other_id"] in {item["source_id"], item["dd_path"]}
                )
            ]
            name["properties"]["source_paths"] = [
                path
                for path in name["properties"]["source_paths"]
                if path not in {item["source_id"], item["dd_path"]}
            ]
            retry = deepcopy(item["retry_event"])
            retry.update(
                {
                    "previous_status": previous["status"],
                    "previous_attempt_count": previous["attempt_count"],
                    "previous_error": previous["last_error"],
                }
            )
            change = deepcopy(item["change_event"])
            source["relationships"].append(
                _recovery_relationship(
                    f"retry-link:{retry['id']}",
                    "HAS_RETRY_EVENT",
                    "out",
                    f"retry-node:{retry['id']}",
                    ["StandardNameSourceRetry"],
                    retry["id"],
                    retry,
                )
            )
            name["relationships"].append(
                _recovery_relationship(
                    f"change-link:{change['id']}",
                    "HAS_INTERNAL_CHANGE",
                    "out",
                    f"change-node:{change['id']}",
                    ["StandardNameChange"],
                    change["id"],
                    change,
                )
            )
            source["properties"].update(
                {
                    "status": "extracted",
                    "produced_sn_id": None,
                    "composed_at": None,
                    "attempt_count": 0,
                    "claimed_at": None,
                    "claim_token": None,
                    "failed_at": None,
                    "last_error": None,
                    "retry_events": previous.get("retry_events", []) + [retry["id"]],
                }
            )
            for relationship in node["relationships"]:
                if relationship["other_id"] == item["source_id"]:
                    relationship["other_properties"] = deepcopy(source["properties"])
            source_ids.append(item["source_id"])
            retry_ids.append(retry["id"])
            change_ids.append(change["id"])
        applied = len(source_ids) - 1 if self.partial else len(source_ids)
        return [
            {
                "applied": applied,
                "source_ids": source_ids[:applied],
                "retry_event_ids": retry_ids[:applied],
                "change_event_ids": change_ids[:applied],
            }
        ]

    def commit(self) -> None:
        self.client.rows = self.working
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _BatchRecoverySession:
    def __init__(self, client: _BatchRecoveryClient) -> None:
        self.client = client

    def __enter__(self):
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def begin_transaction(self) -> _BatchRecoveryTransaction:
        transaction = _BatchRecoveryTransaction(
            self.client,
            race=self.client.race,
            partial=self.client.partial,
        )
        self.client.last_transaction = transaction
        return transaction


class _BatchRecoveryClient:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        race: bool = False,
        partial: bool = False,
    ) -> None:
        self.rows = deepcopy(rows)
        self.race = race
        self.partial = partial
        self.last_transaction: _BatchRecoveryTransaction | None = None

    def session(self) -> _BatchRecoverySession:
        return _BatchRecoverySession(self)


def test_terminal_recovery_manifest_covers_exact_fixture_cohort(
    tmp_path: Path,
) -> None:
    from imas_codex.standard_names.attachment_audit import (
        recover_terminal_attachments,
    )

    rows = [
        _terminal_recovery_closure_row(*fixture)
        for fixture in _TERMINAL_RECOVERY_FIXTURES
    ]
    manifest = tmp_path / "terminal-recovery.json"
    _write_terminal_recovery_manifest(manifest, rows)
    client = _BatchRecoveryClient(rows)

    receipt = recover_terminal_attachments(
        manifest,
        reason="recover exact terminal source bindings",
        gc=client,
    )

    assert receipt["mode"] == "dry_run"
    assert receipt["counts"] == {
        "allowlisted": 9,
        "planned": 9,
        "already_current": 0,
        "applied": 0,
        "refused": 0,
    }
    assert Counter(row["sources"][0]["properties"]["status"] for row in rows) == {
        "composed": 3,
        "attached": 6,
    }
    assert client.rows == rows
    assert client.last_transaction is not None
    assert client.last_transaction.rolled_back is True


def test_terminal_recovery_batch_is_atomic_and_idempotent(tmp_path: Path) -> None:
    from imas_codex.standard_names import attachment_audit as mod

    rows = [
        _terminal_recovery_closure_row(*fixture)
        for fixture in _TERMINAL_RECOVERY_FIXTURES
    ]
    manifest = tmp_path / "terminal-recovery.json"
    manifest_hash = _write_terminal_recovery_manifest(manifest, rows)
    client = _BatchRecoveryClient(rows)

    first = mod.recover_terminal_attachments(
        manifest,
        reason="recover exact terminal source bindings",
        apply=True,
        expected_manifest_hash=manifest_hash,
        gc=client,
    )
    second = mod.recover_terminal_attachments(
        manifest,
        reason="recover exact terminal source bindings",
        apply=True,
        expected_manifest_hash=manifest_hash,
        gc=client,
    )

    assert first["mode"] == "applied"
    assert first["counts"]["applied"] == 9
    assert {row["status"] for row in first["rows"]} == {"applied"}
    assert second["mode"] == "already_current"
    assert second["counts"]["already_current"] == 9
    for row in client.rows:
        source = row["sources"][0]
        name = row["names"][0]
        cohort_size = sum(
            candidate["sn_id"] == row["sn_id"] for candidate in client.rows
        )
        assert source["properties"]["status"] == "extracted"
        assert source["properties"]["produced_sn_id"] is None
        assert row["source_id"] not in name["properties"]["source_paths"]
        assert (
            len(
                [
                    relationship
                    for relationship in source["relationships"]
                    if relationship["type"] == "HAS_RETRY_EVENT"
                ]
            )
            == 1
        )
        assert (
            len(
                [
                    relationship
                    for relationship in name["relationships"]
                    if relationship["type"] == "HAS_INTERNAL_CHANGE"
                    and relationship["other_properties"].get("operation")
                    == "recover_terminal_source_binding"
                ]
            )
            == cohort_size
        )


def test_terminal_recovery_stale_closure_refuses_before_writes(tmp_path: Path) -> None:
    from imas_codex.standard_names.attachment_audit import recover_terminal_attachments

    rows = [_terminal_recovery_closure_row(*_TERMINAL_RECOVERY_FIXTURES[0])]
    manifest = tmp_path / "terminal-recovery.json"
    _write_terminal_recovery_manifest(manifest, rows)
    client = _BatchRecoveryClient(rows)
    client.rows[0]["sources"][0]["properties"]["attempt_count"] = 5
    before = deepcopy(client.rows)

    receipt = recover_terminal_attachments(
        manifest,
        reason="recover exact terminal source binding",
        gc=client,
    )

    assert receipt["mode"] == "refused"
    assert "expected_closure_hash" in " ".join(receipt["refusals"][0]["reasons"])
    assert client.rows == before


def test_terminal_recovery_apply_requires_exact_manifest_hash(tmp_path: Path) -> None:
    from imas_codex.standard_names.attachment_audit import recover_terminal_attachments

    rows = [_terminal_recovery_closure_row(*_TERMINAL_RECOVERY_FIXTURES[0])]
    manifest = tmp_path / "terminal-recovery.json"
    _write_terminal_recovery_manifest(manifest, rows)
    client = _BatchRecoveryClient(rows)

    with pytest.raises(ValueError, match="does not match"):
        recover_terminal_attachments(
            manifest,
            reason="recover exact terminal source binding",
            apply=True,
            expected_manifest_hash="0" * 64,
            gc=client,
        )

    assert client.last_transaction is None


@pytest.mark.parametrize("defect", ["duplicate", "mixed"])
def test_terminal_recovery_manifest_rejects_duplicate_and_mixed_rows(
    tmp_path: Path, defect: str
) -> None:
    from imas_codex.standard_names.attachment_audit import (
        load_terminal_attachment_recovery_manifest,
    )

    rows = [_terminal_recovery_closure_row(*_TERMINAL_RECOVERY_FIXTURES[0])]
    payload = _terminal_recovery_manifest_payload(rows)
    payload["rows"].append(deepcopy(payload["rows"][0]))
    if defect == "mixed":
        payload["rows"][1]["operation"] = "another_operation"
    manifest = tmp_path / "terminal-recovery.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="homogeneous|duplicate|overlapping"):
        load_terminal_attachment_recovery_manifest(manifest)


@pytest.mark.parametrize("defect", ["competing", "west", "fixture"])
def test_terminal_recovery_refuses_unsafe_current_scope(
    tmp_path: Path, defect: str
) -> None:
    from imas_codex.standard_names.attachment_audit import recover_terminal_attachments

    rows = [_terminal_recovery_closure_row(*_TERMINAL_RECOVERY_FIXTURES[0])]
    source = rows[0]["sources"][0]
    if defect == "competing":
        source["relationships"].append(
            _recovery_relationship(
                "binding:competitor",
                "PRODUCED_NAME",
                "out",
                "name:competing",
                ["StandardName"],
                "competing_live_name",
                {"id": "competing_live_name", "name_stage": "accepted"},
            )
        )
    elif defect == "west":
        source["properties"]["facility_id"] = "west"
    else:
        source["properties"]["origin"] = "fixture"
    manifest = tmp_path / "terminal-recovery.json"
    _write_terminal_recovery_manifest(manifest, rows)
    client = _BatchRecoveryClient(rows)

    receipt = recover_terminal_attachments(
        manifest,
        reason="recover exact terminal source binding",
        gc=client,
    )

    assert receipt["mode"] == "refused"
    assert client.last_transaction is not None
    assert client.last_transaction.committed is False


@pytest.mark.parametrize("failure", ["relationship_race", "partial_cardinality"])
def test_terminal_recovery_race_and_partial_apply_roll_back(
    tmp_path: Path, failure: str
) -> None:
    from imas_codex.standard_names.attachment_audit import (
        TerminalAttachmentRecoveryConflict,
        recover_terminal_attachments,
    )

    rows = [_terminal_recovery_closure_row(*_TERMINAL_RECOVERY_FIXTURES[0])]
    manifest = tmp_path / "terminal-recovery.json"
    manifest_hash = _write_terminal_recovery_manifest(manifest, rows)
    client = _BatchRecoveryClient(
        rows,
        race=failure == "relationship_race",
        partial=failure == "partial_cardinality",
    )
    before = deepcopy(client.rows)

    with pytest.raises(TerminalAttachmentRecoveryConflict):
        recover_terminal_attachments(
            manifest,
            reason="recover exact terminal source binding",
            apply=True,
            expected_manifest_hash=manifest_hash,
            gc=client,
        )

    assert client.rows == before
    assert client.last_transaction is not None
    assert client.last_transaction.rolled_back is True
    assert client.last_transaction.committed is False


def _seed_terminal_recovery_graph_cohort(
    scope: _GraphFixtureScope,
    target_sizes: list[int],
) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for target_index, size in enumerate(target_sizes):
        sn_id = scope.uid(f"terminal_target_{target_index}")
        for source_index in range(size):
            dd_path = scope.path(f"terminal/{target_index}/source/{source_index}")
            source_id = _seed_attachment(
                scope,
                dd_path=dd_path,
                sn_id=sn_id,
                name_stage="superseded",
                dd_unit="1",
                sn_unit="1",
                source_status="attached" if source_index % 2 else "composed",
            )
            pairs.append({"source_id": source_id, "dd_path": dd_path, "sn_id": sn_id})
        unit_id = scope.uid(f"unit_{target_index}")
        review_id = scope.uid(f"review_{target_index}")
        revision_id = scope.uid(f"revision_{target_index}")
        predecessor_id = scope.uid(f"predecessor_{target_index}")
        scope.track("Unit", unit_id)
        scope.track("Review", review_id)
        scope.track("DocsRevision", revision_id)
        scope.track("StandardName", predecessor_id)
        scope.gc.query(
            """
            MATCH (name:StandardName {id: $sn_id})
            SET name.protected_marker = 'preserved',
                name.reviewer_score_name = 0.93,
                name.reviewer_score_docs = 0.95,
                name.cocos = 17
            MERGE (unit:Unit {id: $unit_id}) SET unit.symbol = '1'
            MERGE (review:Review {id: $review_id}) SET review.score = 0.93
            MERGE (revision:DocsRevision {id: $revision_id})
              SET revision.description = 'preserved revision'
            MERGE (predecessor:StandardName {id: $predecessor_id})
              SET predecessor.name_stage = 'superseded'
            MERGE (name)-[:HAS_UNIT]->(unit)
            MERGE (name)-[:HAS_REVIEW]->(review)
            MERGE (revision)-[:DOCS_REVISION_OF]->(name)
            MERGE (name)-[:REFINED_FROM]->(predecessor)
            """,
            sn_id=sn_id,
            unit_id=unit_id,
            review_id=review_id,
            revision_id=revision_id,
            predecessor_id=predecessor_id,
        )
    return sorted(pairs, key=lambda item: item["source_id"])


def _write_graph_terminal_recovery_manifest(
    gc: Any,
    path: Path,
    pairs: list[dict[str, str]],
) -> str:
    from imas_codex.standard_names import attachment_audit as mod

    with gc.session() as session:
        transaction = session.begin_transaction()
        try:
            rows = mod._read_terminal_recovery_rows(transaction, tuple(pairs))
        finally:
            transaction.rollback()
    return _write_terminal_recovery_manifest(path, rows)


def _terminal_recovery_graph_counts(gc: Any, pairs: list[dict[str, str]]) -> dict:
    source_ids = [pair["source_id"] for pair in pairs]
    dd_paths = [pair["dd_path"] for pair in pairs]
    sn_ids = sorted({pair["sn_id"] for pair in pairs})
    rows = gc.query(
        """
        MATCH (source:StandardNameSource)
        WHERE source.id IN $source_ids
        OPTIONAL MATCH (source)-[binding:PRODUCED_NAME]->(:StandardName)
        OPTIONAL MATCH (source)-[:HAS_RETRY_EVENT]->(retry:StandardNameSourceRetry)
        WITH count(DISTINCT binding) AS bindings,
             count(DISTINCT retry) AS retries
        MATCH (node:IMASNode)
        WHERE node.id IN $dd_paths
        OPTIONAL MATCH (node)-[projection:HAS_STANDARD_NAME]->(:StandardName)
        WITH bindings, retries, count(DISTINCT projection) AS projections
        MATCH (name:StandardName)
        WHERE name.id IN $sn_ids
        OPTIONAL MATCH (name)-[:HAS_INTERNAL_CHANGE]->(change:StandardNameChange)
        OPTIONAL MATCH (name)-[:HAS_REVIEW]->(review:Review)
        OPTIONAL MATCH (revision:DocsRevision)-[:DOCS_REVISION_OF]->(name)
        OPTIONAL MATCH (name)-[:REFINED_FROM]->(predecessor:StandardName)
        OPTIONAL MATCH (name)-[:HAS_UNIT]->(unit:Unit)
        RETURN bindings, projections, retries,
               count(DISTINCT change) AS changes,
               count(DISTINCT review) AS reviews,
               count(DISTINCT revision) AS revisions,
               count(DISTINCT predecessor) AS predecessors,
               count(DISTINCT unit) AS units,
               collect(DISTINCT name.protected_marker) AS protected_markers
        """,
        source_ids=source_ids,
        dd_paths=dd_paths,
        sn_ids=sn_ids,
    )
    return dict(rows[0])


class _TerminalRecoveryInjectedTransaction:
    def __init__(
        self,
        transaction: Any,
        *,
        failure: str,
        source_id: str,
        target_id: str,
    ) -> None:
        self.transaction = transaction
        self.failure = failure
        self.source_id = source_id
        self.target_id = target_id
        self.injected = False

    def run(self, query: str, **params: Any):
        if (
            "TERMINAL_ATTACHMENT_RECOVERY_RELATIONSHIP_LOCK" in query
            and self.failure == "relationship_drift"
            and not self.injected
        ):
            list(
                self.transaction.run(
                    """
                    MATCH (source:StandardNameSource {id: $source_id})
                          -[binding:PRODUCED_NAME]->
                          (name:StandardName {id: $target_id})
                    SET binding.concurrent_marker = 'drifted'
                    RETURN count(binding) AS changed
                    """,
                    source_id=self.source_id,
                    target_id=self.target_id,
                )
            )
            self.injected = True
            return self.transaction.run(query, **params)
        if (
            "TERMINAL_ATTACHMENT_RECOVERY_APPLY" in query
            and self.failure == "mid_batch"
        ):
            list(self.transaction.run(query, **params))
            raise RuntimeError("injected mid-batch failure")
        return self.transaction.run(query, **params)

    def commit(self) -> None:
        self.transaction.commit()

    def rollback(self) -> None:
        self.transaction.rollback()


class _TerminalRecoveryInjectedSession:
    def __init__(
        self,
        session: Any,
        *,
        failure: str,
        source_id: str,
        target_id: str,
    ) -> None:
        self.session = session
        self.failure = failure
        self.source_id = source_id
        self.target_id = target_id

    def begin_transaction(self) -> _TerminalRecoveryInjectedTransaction:
        return _TerminalRecoveryInjectedTransaction(
            self.session.begin_transaction(),
            failure=self.failure,
            source_id=self.source_id,
            target_id=self.target_id,
        )


class _TerminalRecoveryInjectedClient:
    def __init__(
        self,
        gc: Any,
        *,
        failure: str,
        source_id: str,
        target_id: str,
    ) -> None:
        self.gc = gc
        self.failure = failure
        self.source_id = source_id
        self.target_id = target_id

    @contextmanager
    def session(self):
        with self.gc.session() as session:
            yield _TerminalRecoveryInjectedSession(
                session,
                failure=self.failure,
                source_id=self.source_id,
                target_id=self.target_id,
            )


@pytest.mark.graph
def test_terminal_recovery_relationship_lock_plan_is_participant_anchored(
    _gc: Any,
) -> None:
    from imas_codex.standard_names import attachment_audit as mod

    descriptor = {
        "element_id": "4:terminal-recovery-relationship:0",
        "owner_element_id": "4:terminal-recovery-owner:0",
        "other_element_id": "4:terminal-recovery-other:0",
        "type": "PRODUCED_NAME",
        "direction": "out",
    }
    with _gc.session() as session:
        plan = (
            session.run(
                f"EXPLAIN {mod._TERMINAL_RECOVERY_RELATIONSHIP_LOCK_QUERY}",
                relationships=[descriptor],
            )
            .consume()
            .plan
        )

    assert plan is not None
    pending = [plan]
    operators: list[str] = []
    while pending:
        operator = pending.pop()
        if isinstance(operator, dict):
            operator_type = str(operator["operatorType"]).partition("@")[0]
            children = operator.get("children") or []
        else:
            operator_type = operator.operator_type
            children = operator.children
        operators.append(operator_type)
        pending.extend(children)

    forbidden = {
        "AllRelationshipsScan",
        "DirectedAllRelationshipsScan",
        "UndirectedAllRelationshipsScan",
    }
    assert forbidden.isdisjoint(operators), operators
    assert any(
        operator == "NodeByElementIdSeek" or "NodeIndexSeek" in operator
        for operator in operators
    ), operators
    assert any(operator.startswith("Expand(") for operator in operators), operators


@pytest.mark.graph
@pytest.mark.parametrize("shared_count", [2, 6])
def test_terminal_recovery_graph_shared_target_is_atomic_and_idempotent(
    _gc: Any,
    _clean: _GraphFixtureScope,
    tmp_path: Path,
    shared_count: int,
) -> None:
    from imas_codex.standard_names.attachment_audit import recover_terminal_attachments

    pairs = _seed_terminal_recovery_graph_cohort(_clean, [shared_count])
    manifest = tmp_path / "terminal-recovery.json"
    manifest_hash = _write_graph_terminal_recovery_manifest(_gc, manifest, pairs)

    first = recover_terminal_attachments(
        manifest,
        reason="recover exact shared terminal bindings",
        apply=True,
        expected_manifest_hash=manifest_hash,
        gc=_gc,
    )
    second = recover_terminal_attachments(
        manifest,
        reason="recover exact shared terminal bindings",
        apply=True,
        expected_manifest_hash=manifest_hash,
        gc=_gc,
    )
    counts = _terminal_recovery_graph_counts(_gc, pairs)

    assert first["mode"] == "applied"
    assert first["counts"]["applied"] == shared_count
    assert {row["status"] for row in first["rows"]} == {"applied"}
    assert second["mode"] == "already_current"
    assert second["counts"]["already_current"] == shared_count
    assert counts == {
        "bindings": 0,
        "projections": 0,
        "retries": shared_count,
        "changes": shared_count,
        "reviews": 1,
        "revisions": 1,
        "predecessors": 1,
        "units": 1,
        "protected_markers": ["preserved"],
    }


@pytest.mark.graph
def test_terminal_recovery_graph_mixed_unique_and_shared_targets(
    _gc: Any, _clean: _GraphFixtureScope, tmp_path: Path
) -> None:
    from imas_codex.standard_names.attachment_audit import recover_terminal_attachments

    pairs = _seed_terminal_recovery_graph_cohort(_clean, [2, 1, 1])
    manifest = tmp_path / "terminal-recovery.json"
    manifest_hash = _write_graph_terminal_recovery_manifest(_gc, manifest, pairs)

    receipt = recover_terminal_attachments(
        manifest,
        reason="recover mixed exact terminal bindings",
        apply=True,
        expected_manifest_hash=manifest_hash,
        gc=_gc,
    )

    assert receipt["mode"] == "applied"
    assert receipt["counts"]["applied"] == 4
    counts = _terminal_recovery_graph_counts(_gc, pairs)
    assert counts["bindings"] == counts["projections"] == 0
    assert counts["retries"] == counts["changes"] == 4
    assert counts["reviews"] == counts["revisions"] == 3


@pytest.mark.graph
def test_terminal_recovery_graph_stale_sibling_refuses_before_writes(
    _gc: Any, _clean: _GraphFixtureScope, tmp_path: Path
) -> None:
    from imas_codex.standard_names.attachment_audit import recover_terminal_attachments

    pairs = _seed_terminal_recovery_graph_cohort(_clean, [2])
    manifest = tmp_path / "terminal-recovery.json"
    manifest_hash = _write_graph_terminal_recovery_manifest(_gc, manifest, pairs)
    _gc.query(
        "MATCH (name:StandardName {id: $sn_id}) SET name.protected_marker = 'stale'",
        sn_id=pairs[0]["sn_id"],
    )

    receipt = recover_terminal_attachments(
        manifest,
        reason="recover exact shared terminal bindings",
        apply=True,
        expected_manifest_hash=manifest_hash,
        gc=_gc,
    )
    counts = _terminal_recovery_graph_counts(_gc, pairs)

    assert receipt["mode"] == "refused"
    assert counts["bindings"] == counts["projections"] == 2
    assert counts["retries"] == counts["changes"] == 0


@pytest.mark.graph
@pytest.mark.parametrize(
    ("failure", "expected_exception"),
    [
        ("relationship_drift", "TerminalAttachmentRecoveryConflict"),
        ("mid_batch", "RuntimeError"),
    ],
)
def test_terminal_recovery_graph_races_roll_back_complete_cohort(
    _gc: Any,
    _clean: _GraphFixtureScope,
    tmp_path: Path,
    failure: str,
    expected_exception: str,
) -> None:
    from imas_codex.standard_names.attachment_audit import (
        TerminalAttachmentRecoveryConflict,
        recover_terminal_attachments,
    )

    pairs = _seed_terminal_recovery_graph_cohort(_clean, [2])
    manifest = tmp_path / "terminal-recovery.json"
    manifest_hash = _write_graph_terminal_recovery_manifest(_gc, manifest, pairs)
    client = _TerminalRecoveryInjectedClient(
        _gc,
        failure=failure,
        source_id=pairs[0]["source_id"],
        target_id=pairs[0]["sn_id"],
    )
    exception_type = (
        TerminalAttachmentRecoveryConflict
        if expected_exception == "TerminalAttachmentRecoveryConflict"
        else RuntimeError
    )

    with pytest.raises(exception_type):
        recover_terminal_attachments(
            manifest,
            reason="recover exact shared terminal bindings",
            apply=True,
            expected_manifest_hash=manifest_hash,
            gc=client,
        )
    counts = _terminal_recovery_graph_counts(_gc, pairs)

    assert counts["bindings"] == counts["projections"] == 2
    assert counts["retries"] == counts["changes"] == 0


@pytest.mark.graph
def test_terminal_recovery_graph_fixture_refuses_production_uri() -> None:
    from imas_codex.settings import get_graph_uri

    test_uri = os.environ["IMAS_CODEX_TEST_NEO4J_URI"]
    project_uri = os.environ.get("IMAS_CODEX_TEST_PROJECT_NEO4J_URI") or get_graph_uri()

    assert os.environ["IMAS_CODEX_TEST_NEO4J_EPHEMERAL"] == "1"
    assert test_uri != project_uri


# ---------------------------------------------------------------------------
# Ordinary targeted detach — semantic mis-shares on live names
# ---------------------------------------------------------------------------


def _detach_client(
    *,
    exists: bool = True,
    other_live: int = 0,
    name_attachments: int = 2,
    projection_only: bool = False,
    structural_parent: bool = False,
) -> MagicMock:
    """A client answering the targeted detach's pre-flight read."""
    from imas_codex.standard_names import attachment_audit as mod

    def _query(q: str, **params):
        if "RETURN src.id AS source_node_id" in q:
            if not exists:
                return [
                    {
                        "source_node_id": None,
                        "other_live_names": 0,
                        "projected": False,
                        "name_attachments": 0,
                        "structural_parent": False,
                    }
                ]
            return [
                {
                    "source_node_id": None if projection_only else "dd:some/path",
                    "other_live_names": other_live,
                    "projected": True,
                    "name_attachments": name_attachments,
                    "structural_parent": structural_parent,
                }
            ]
        if q in (mod._DETACH_QUERY, mod._DETACH_PROJECTION_QUERY):
            return [{"detached": 1}]
        return []

    gc = MagicMock()
    gc.query.side_effect = _query
    return gc


def test_detach_removes_a_semantic_mis_share() -> None:
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc = _detach_client(other_live=1)
    res = detach_one_attachment(
        "spectrometer_visible/channel/grating_spectrometer/radiance_spectral",
        "spectral_bremsstrahlung_radiance",
        reason="line emission, not continuum",
        gc=gc,
    )
    assert res["ok"] is True
    assert res["source_rewound"] is False, "the source still has a live name"


def test_detach_rewinds_a_source_left_with_no_live_name() -> None:
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc = _detach_client(other_live=0)
    res = detach_one_attachment("a/b", "some_name", reason="wrong", gc=gc)
    assert res["ok"] is True and res["source_rewound"] is True


def test_detach_refuses_to_orphan_a_name() -> None:
    """A name every source rejects is a NAME defect, not an attachment defect."""
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc = _detach_client(name_attachments=1)
    res = detach_one_attachment("a/b", "only_here", reason="wrong", gc=gc)
    assert res["ok"] is False
    assert "sn edit --rename" in res["reason"]


def test_detach_releases_a_derived_parent_holding_its_childs_leaf() -> None:
    """A derived parent with live children is anchored by structure, not sources.

    Accepted derived parents routinely carry no realization at all, so losing
    the last one returns the parent to that designed state — the would-orphan
    refusal must not fire.
    """
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc = _detach_client(name_attachments=1, other_live=2, structural_parent=True)
    res = detach_one_attachment(
        "summary/global_quantities/greenwald_fraction/value",
        "greenwald_density",
        reason="the leaf is the dimensionless fraction; the parent is its ratio operand",
        gc=gc,
    )
    assert res["ok"] is True
    assert res["structural_parent"] is True
    assert res["source_rewound"] is False, "the path keeps other live names"


def test_detach_refuses_a_pairing_that_does_not_exist() -> None:
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc = _detach_client(exists=False)
    res = detach_one_attachment("a/b", "unrelated", reason="wrong", gc=gc)
    assert res["ok"] is False and "does not realize" in res["reason"]


def test_detach_reaches_a_projection_with_no_provenance() -> None:
    """A DD-side realization can exist with no StandardNameSource behind it.

    The export reads that projection, so it is exactly the kind of pairing that
    reaches the catalog — and there is no source to rewind, only the dangling
    assertion to remove.
    """
    from imas_codex.standard_names import attachment_audit as mod
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc = _detach_client(projection_only=True)
    res = detach_one_attachment("a/b", "some_name", reason="wrong sensor", gc=gc)
    assert res["ok"] is True
    assert res["projection_only"] is True
    assert res["source_rewound"] is False, "there is no source to rewind"
    assert any(
        call.args[0] == mod._DETACH_PROJECTION_QUERY for call in gc.query.call_args_list
    )


def test_detach_dry_run_does_not_write() -> None:
    from imas_codex.standard_names import attachment_audit as mod
    from imas_codex.standard_names.attachment_audit import detach_one_attachment

    gc = _detach_client()
    res = detach_one_attachment("a/b", "n", reason="wrong", gc=gc, dry_run=True)
    assert res["ok"] is True and res["dry_run"] is True
    assert all(call.args[0] != mod._DETACH_QUERY for call in gc.query.call_args_list), (
        "dry run issued the detach write"
    )


def test_gate_survives_a_graph_failure() -> None:
    """A gate that raised would turn a bad edge into a lost rename."""
    from imas_codex.standard_names.attachment_audit import gate_migrated_attachments

    gc = MagicMock()
    gc.query.side_effect = RuntimeError("graph unavailable")
    result = gate_migrated_attachments(gc, sn_id="electron_density")
    assert result == AttachmentAuditResult()


@pytest.mark.graph
def test_orphaned_name_is_reported(_gc, _clean):
    """A name whose only source was wrong is flagged, not auto-superseded."""
    dd_path = _clean.path("summary/boundary/strike_point_inner_z/value")
    sn_id = _clean.uid("z_image_up_unit_vector_of_camera")
    _seed_attachment(_clean, dd_path=dd_path, sn_id=sn_id)

    result = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))
    assert result.names_orphaned == [sn_id]

    still_there = _gc.query(
        "MATCH (sn:StandardName {id: $sn}) RETURN sn.name_stage AS stage", sn=sn_id
    )
    assert still_there[0]["stage"] == "drafted", "the name must not be auto-retired"


@pytest.mark.graph
def test_conflicting_group_leaves_one_edge_standing(_gc, _clean):
    """Only the surplus member of a pairwise-conflicting group is detached."""
    kept = _clean.path("camera_ir/channel/camera/direction/z")
    surplus = _clean.path("camera_ir/channel/camera/up/z")
    sn_id = _clean.uid("z_direction_unit_vector_of_camera")
    for path in (surplus, kept):  # seeded surplus-first on purpose
        _seed_attachment(_clean, dd_path=path, sn_id=sn_id, dd_unit="1", sn_unit="1")

    result = reconcile_attachment_consistency(_ScopedClient(_gc, _clean.prefix))
    assert [v.dd_path for v in result.rejected] == [surplus]
    assert result.detached == 1
    assert result.names_orphaned == [], "the representative keeps the name sourced"

    rows = _gc.query(
        """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $sn})
        MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
        RETURN collect(dd.id) AS paths, sn.source_paths AS source_paths
        """,
        sn=sn_id,
    )
    assert rows[0]["paths"] == [kept]
    assert rows[0]["source_paths"] == [f"dd:{kept}"]


@pytest.mark.graph
def test_name_level_defect_keeps_every_attachment(_gc, _clean):
    """A uniformly-rejected name is handed to ``sn edit``, not stripped."""
    sn_id = _clean.uid("atomic_count_of_ion_state")
    paths = [
        _clean.path("core_profiles/profiles_1d/ion/element/atoms_n"),
        _clean.path("edge_profiles/profiles_1d/ion/element/atoms_n"),
    ]
    for path in paths:
        _seed_attachment(
            _clean,
            dd_path=path,
            sn_id=sn_id,
            name_stage="accepted",
            dd_unit="1",
            sn_unit="1",
        )

    result = reconcile_attachment_consistency(
        _ScopedClient(_gc, _clean.prefix), include_accepted=True
    )
    assert len(result.rejected) == 2
    assert [d.sn_id for d in result.names_misnamed] == [sn_id]
    assert result.skipped_misnamed == 2
    assert result.detached == 0

    rows = _gc.query(
        """
        MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName {id: $sn})
        MATCH (src)-[:FROM_DD_PATH]->(dd:IMASNode)
        RETURN count(*) AS attachments, collect(src.status) AS statuses
        """,
        sn=sn_id,
    )
    assert rows[0]["attachments"] == 2, "no attachment may be detached"
    assert set(rows[0]["statuses"]) == {"composed"}, "no source may be rewound"
