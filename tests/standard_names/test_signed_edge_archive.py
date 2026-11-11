"""Signed archival of the edges that close a refined-from lineage cycle.

A cycle in ``REFINED_FROM`` has to be broken without moving the lineage of any
live identity. The operation these tests drive archives the exact edges a
signed manifest names, mints one receipt per edge carrying the full edge
snapshot, and removes nothing else. The planted graph is the disposable
endpoint the standard-names suite starts, so ``GraphClient`` here never
reaches the shared graph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.signed_manifest import (
    SignedManifestAuthorityError,
    apply_signed_manifest,
    compose_edge_archive_authority,
    signed_payload_sha256,
)

pytestmark = pytest.mark.graph

_ADAPTER = "edge-archive"
_MUTATION = "archive-lineage-cycle-edges"
_GUARDS = (
    "signed-edge-snapshot-manifest",
    "component-edge-closure",
    "out-of-allowlist-immutability",
)
_REASON = "break a refined-from lineage cycle at its superseded members"


def _seed(nodes: dict[str, str], edges: list[tuple[str, str]]) -> None:
    """Create names with their lifecycle stage and the given lineage edges."""
    with GraphClient() as gc:
        gc.query("MATCH (node) DETACH DELETE node")
        gc.query(
            """
            UNWIND $nodes AS row
            CREATE (:StandardName {
              id: row.id, name_stage: row.stage,
              status: CASE WHEN row.stage = 'superseded' THEN 'superseded'
                           ELSE 'draft' END,
              origin: 'pipeline'})
            """,
            nodes=[{"id": name, "stage": stage} for name, stage in nodes.items()],
        )
        if edges:
            gc.query(
                """
                UNWIND $edges AS row
                MATCH (successor:StandardName {id: row.successor}),
                      (predecessor:StandardName {id: row.predecessor})
                CREATE (successor)-[:REFINED_FROM]->(predecessor)
                """,
                edges=[
                    {"successor": successor, "predecessor": predecessor}
                    for successor, predecessor in edges
                ],
            )


def _live_edges() -> list[tuple[str, str]]:
    with GraphClient() as gc:
        rows = gc.query(
            """MATCH (a:StandardName)-[:REFINED_FROM]->(b:StandardName)
            RETURN a.id AS successor, b.id AS predecessor
            ORDER BY successor, predecessor"""
        )
    return [(row["successor"], row["predecessor"]) for row in rows]


def _lifecycle(name_id: str) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """MATCH (node:StandardName {id: $id})
            RETURN node.name_stage AS stage, node.status AS status""",
            id=name_id,
        )
    assert len(rows) < 2
    return dict(rows[0]) if rows else {}


def _snapshot(name_id: str, predecessor: str, component: dict) -> dict:
    with GraphClient() as gc:
        rows = gc.query(
            """MATCH (successor:StandardName {id: $successor})
                     -[edge:REFINED_FROM]->
                     (predecessor:StandardName {id: $predecessor})
            RETURN elementId(edge) AS element_id, properties(edge) AS properties,
                   successor.name_stage AS successor_stage,
                   successor.status AS successor_status,
                   predecessor.name_stage AS predecessor_stage,
                   predecessor.status AS predecessor_status""",
            successor=name_id,
            predecessor=predecessor,
        )
    assert len(rows) == 1
    row = dict(rows[0])
    return {
        "successor": name_id,
        "predecessor": predecessor,
        "relationship_type": "REFINED_FROM",
        "component": component["component"],
        "element_id": row["element_id"],
        "properties": dict(row["properties"] or {}),
        "successor_stage": row["successor_stage"],
        "successor_status": row["successor_status"],
        "predecessor_stage": row["predecessor_stage"],
        "predecessor_status": row["predecessor_status"],
    }


def _component(members: list[str], live_members: list[str], rule: str) -> dict:
    return {
        "component": "component-"
        + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()[:12],
        "members": members,
        "live_members": live_members,
        "rule": rule,
    }


def _write_authority(
    tmp_path: Path, components: list[dict], edges: list[dict]
) -> tuple[Path, str, str]:
    authority = compose_edge_archive_authority(
        operation_id="test-lineage-cycle-repair",
        components=components,
        edges=edges,
    )
    path = tmp_path / "edge-archive-authority.json"
    raw = json.dumps(authority, sort_keys=True).encode()
    path.write_bytes(raw)
    return (
        path,
        hashlib.sha256(raw).hexdigest(),
        signed_payload_sha256(json.loads(raw)),
    )


def _apply(path: Path, file_sha256: str, payload_sha256: str, **kwargs):
    return apply_signed_manifest(
        path,
        authority_file_sha256=file_sha256,
        authority_payload_sha256=payload_sha256,
        authority_adapter=_ADAPTER,
        mutation_kind=_MUTATION,
        guard_set=_GUARDS,
        reason=_REASON,
        **kwargs,
    )


def _preview_and_apply(path: Path, file_sha256: str, payload_sha256: str):
    """Preview the signed closure, then authorize it with its own digest."""
    preview = _apply(path, file_sha256, payload_sha256, apply=False)
    assert preview["outcome"] == "would_apply", preview["refusals"]
    return _apply(
        path,
        file_sha256,
        payload_sha256,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )


def _receipts() -> list[dict]:
    with GraphClient() as gc:
        rows = gc.query(
            """MATCH (change:StandardNameChange)
            RETURN properties(change) AS properties ORDER BY change.id"""
        )
    return [dict(row["properties"]) for row in rows]


def test_archives_the_edge_the_manifest_names_and_keeps_the_live_name(tmp_path):
    """A single-live cycle loses its superseded member's ancestry edge only."""
    _seed(
        {"cycle_parent": "superseded", "cycle_child": "accepted"},
        [
            ("cycle_parent", "cycle_child"),
            ("cycle_child", "cycle_parent"),
        ],
    )
    with GraphClient() as gc:
        gc.query(
            """MATCH (child:StandardName {id: 'cycle_child'})
            CREATE (source:StandardNameSource {
              id: 'dd:test/cycle/child', source_type: 'dd',
              source_id: 'test/cycle/child', status: 'attached'})
            CREATE (source)-[:PRODUCED_NAME]->(child)
            CREATE (child)-[:HAS_REVIEW]->(:StandardNameReview {id: 'review-cycle'})
            CREATE (child)-[:DOCS_REVISION_OF]->(:DocsRevision {id: 'docs-cycle'})"""
        )
    component = _component(
        ["cycle_parent", "cycle_child"], ["cycle_child"], "single-live"
    )
    path, file_sha256, payload_sha256 = _write_authority(
        tmp_path,
        [component],
        [_snapshot("cycle_parent", "cycle_child", component)],
    )

    result = _preview_and_apply(path, file_sha256, payload_sha256)

    assert result["outcome"] == "applied"
    assert result["changed"] == 1
    assert _live_edges() == [("cycle_child", "cycle_parent")]
    assert _lifecycle("cycle_child")["stage"] == "accepted"
    assert _lifecycle("cycle_parent")["stage"] == "superseded"
    assert result["postcondition"][0]["members"] == {
        "cycle_child": {
            "name_stage": "accepted",
            "status": "draft",
            "sources": 1,
            "reviews": 1,
            "documents": 1,
        },
        "cycle_parent": {
            "name_stage": "superseded",
            "status": "superseded",
            "sources": 0,
            "reviews": 0,
            "documents": 0,
        },
    }
    receipts = _receipts()
    assert len(receipts) == 1
    snapshot = json.loads(receipts[0]["edge_snapshot"])
    assert snapshot["successor"] == "cycle_parent"
    assert snapshot["predecessor"] == "cycle_child"
    assert snapshot["relationship_type"] == "REFINED_FROM"
    assert snapshot["element_id"]
    assert receipts[0]["operation"] == "archive_lineage_cycle_edges"
    assert receipts[0]["origin"] == "signed_manifest"


def test_archives_every_internal_edge_of_a_cycle_with_no_live_member(tmp_path):
    """A retired cycle has all of its internal edges archived."""
    _seed(
        {"retired_one": "superseded", "retired_two": "superseded"},
        [
            ("retired_one", "retired_two"),
            ("retired_two", "retired_one"),
        ],
    )
    component = _component(["retired_one", "retired_two"], [], "no-live")
    path, file_sha256, payload_sha256 = _write_authority(
        tmp_path,
        [component],
        [
            _snapshot("retired_one", "retired_two", component),
            _snapshot("retired_two", "retired_one", component),
        ],
    )

    result = _preview_and_apply(path, file_sha256, payload_sha256)

    assert result["outcome"] == "applied"
    assert result["changed"] == 2
    assert _live_edges() == []
    assert len(_receipts()) == 2


def test_refuses_an_edge_inside_the_component_that_the_manifest_omits(tmp_path):
    """An undeclared live edge cannot be swept into the same archive."""
    _seed(
        {
            "leak_live": "accepted",
            "leak_superseded": "superseded",
            "leak_other": "superseded",
        },
        [
            ("leak_superseded", "leak_live"),
            ("leak_live", "leak_superseded"),
            ("leak_other", "leak_live"),
        ],
    )
    component = _component(
        ["leak_live", "leak_superseded", "leak_other"],
        ["leak_live"],
        "single-live",
    )
    path, file_sha256, payload_sha256 = _write_authority(
        tmp_path,
        [component],
        [_snapshot("leak_superseded", "leak_live", component)],
    )

    result = _apply(
        path,
        file_sha256,
        payload_sha256,
        apply=True,
        manifest_sha256="0" * 64,
    )

    assert result["outcome"] == "refused"
    assert result["changed"] == 0
    assert any(
        "outside the manifest" in refusal["reason"] for refusal in result["refusals"]
    )
    assert len(_live_edges()) == 3
    assert _receipts() == []


def test_preview_writes_nothing_and_reports_the_signed_manifest(tmp_path):
    _seed(
        {"preview_child": "accepted", "preview_parent": "superseded"},
        [
            ("preview_parent", "preview_child"),
            ("preview_child", "preview_parent"),
        ],
    )
    component = _component(
        ["preview_parent", "preview_child"], ["preview_child"], "single-live"
    )
    path, file_sha256, payload_sha256 = _write_authority(
        tmp_path,
        [component],
        [_snapshot("preview_parent", "preview_child", component)],
    )

    result = _apply(path, file_sha256, payload_sha256, apply=False)

    assert result["outcome"] == "would_apply"
    assert result["would_change"] == 1
    assert result["manifest_sha256"]
    assert len(_live_edges()) == 2
    assert _receipts() == []


def test_apply_requires_the_authorized_manifest_sha256(tmp_path):
    _seed(
        {"stale_child": "accepted", "stale_parent": "superseded"},
        [
            ("stale_parent", "stale_child"),
            ("stale_child", "stale_parent"),
        ],
    )
    component = _component(
        ["stale_parent", "stale_child"], ["stale_child"], "single-live"
    )
    path, file_sha256, payload_sha256 = _write_authority(
        tmp_path,
        [component],
        [_snapshot("stale_parent", "stale_child", component)],
    )

    preview = _apply(path, file_sha256, payload_sha256, apply=False)
    with pytest.raises(Exception) as refusal:
        _apply(
            path,
            file_sha256,
            payload_sha256,
            apply=True,
            manifest_sha256="0" * 64,
        )
    assert "does not match authorized" in str(refusal.value)
    assert len(_live_edges()) == 2

    replayed = _apply(
        path,
        file_sha256,
        payload_sha256,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert replayed["outcome"] == "applied"
    again = _apply(
        path,
        file_sha256,
        payload_sha256,
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )
    assert again["outcome"] == "replayed"
    assert len(_receipts()) == 1


def test_refuses_an_authority_whose_edges_are_outside_a_declared_component():
    with pytest.raises(SignedManifestAuthorityError):
        compose_edge_archive_authority(
            operation_id="test-lineage-cycle-repair",
            components=[_component(["a", "b"], ["b"], "single-live")],
            edges=[
                {
                    "successor": "a",
                    "predecessor": "c",
                    "relationship_type": "REFINED_FROM",
                    "element_id": "4:abc:1",
                    "component": "not-declared",
                    "properties": {},
                }
            ],
        )
