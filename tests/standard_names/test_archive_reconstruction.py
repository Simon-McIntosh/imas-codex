"""Closed signed reconstruction for archived standard-name identities."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names.signed_manifest import (
    SignedManifestAuthorityError,
    SignedManifestConflict,
    apply_signed_manifest,
    signed_payload_sha256,
)

_ADAPTER = "archive-reconstruction"
_MUTATION = "reconstruct-archived-standard-names"
_GUARDS = (
    "signed-exact-identity-manifest",
    "allowlisted-archive-relationships",
    "archive-live-edge-count-parity",
)


class _Transaction:
    def __init__(self, graph: _ArchiveGraph) -> None:
        self.graph = graph
        self.closed = False

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "archive-reconstruction-node-state" in cypher:
            node = self.graph.nodes.get(params["id"])
            return (
                [] if node is None else [{"properties": deepcopy(node["properties"])}]
            )
        if "archive-reconstruction-counterpart-state" in cypher:
            label = re.search(r"MATCH \(counterpart:([A-Za-z]+)", cypher).group(1)
            counterpart = self.graph.counterparts.get((label, params["id"]))
            return (
                []
                if counterpart is None
                else [{"properties": deepcopy(counterpart["properties"])}]
            )
        if "archive-reconstruction-create-node" in cypher:
            node = self.graph.nodes.setdefault(
                params["id"], {"properties": deepcopy(params["properties"])}
            )
            return [{"count": int(node["properties"] == params["properties"])}]
        if "archive-reconstruction-create-counterpart" in cypher:
            label = re.search(r"MERGE \(node:([A-Za-z]+)", cypher).group(1)
            counterpart = self.graph.counterparts.setdefault(
                (label, params["id"]), {"properties": deepcopy(params["properties"])}
            )
            return [{"count": int(counterpart["properties"] == params["properties"])}]
        if "archive-reconstruction-create-edge" in cypher:
            relationship_type = re.search(r"\[relationship:([A-Z_]+)\]", cypher).group(
                1
            )
            edge_key = (params["start_id"], relationship_type, params["end_id"])
            for index, edge in enumerate(self.graph.edges):
                if edge[:3] == edge_key:
                    properties = deepcopy(edge[3])
                    properties.update(deepcopy(params["properties"]))
                    self.graph.edges[index] = (*edge_key, properties)
                    return [{"count": 1}]
            self.graph.edges.append((*edge_key, deepcopy(params["properties"])))
            return [{"count": 1}]
        if "archive-reconstruction-edge-counts" in cypher:
            return [
                {
                    "relationship_type": relationship_type,
                    "count": sum(
                        1
                        for start, edge_type, end, _ in self.graph.edges
                        if edge_type == relationship_type
                        and params["id"] in {start, end}
                    )
                    + self.graph.extra_counts.get((params["id"], relationship_type), 0),
                }
                for relationship_type in params["types"]
            ]
        raise AssertionError(f"unexpected query: {cypher}")

    def rollback(self) -> None:
        self.closed = True

    def commit(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, graph: _ArchiveGraph) -> None:
        self.graph = graph

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def begin_transaction(self) -> _Transaction:
        return _Transaction(self.graph)


class _ArchiveGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.counterparts = {("Unit", "unit:eV"): {"properties": {"id": "unit:eV"}}}
        self.edges: list[tuple[str, str, str, dict[str, Any]]] = []
        self.extra_counts: dict[tuple[str, str], int] = {}

    def session(self) -> _Session:
        return _Session(self)


def _authority(
    edges: list[dict[str, Any]] | None = None,
    counterparts: list[dict[str, Any]] | None = None,
    archive_roles: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    authority = {
        "schema": "imas-codex.archive-reconstruction.v1",
        "operation_id": "reconstruct_archived_standard_name",
        "identities": ["archived_temperature"],
        "nodes": [
            {
                "id": "archived_temperature",
                "properties": {
                    "id": "archived_temperature",
                    "origin": "catalog_edit",
                    "status": "draft",
                    "name_stage": "accepted",
                },
            }
        ],
        "edges": edges
        if edges is not None
        else [
            {
                "owner_id": "archived_temperature",
                "relationship_type": "HAS_UNIT",
                "direction": "outgoing",
                "counterpart_id": "unit:eV",
                "properties": {"source": "archive"},
            }
        ],
    }
    if counterparts is not None:
        authority["counterparts"] = counterparts
    if archive_roles is not None:
        authority["archive_roles"] = archive_roles
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": signed_payload_sha256(authority),
    }
    return authority


def _write_authority(path: Path, authority: dict[str, Any]) -> tuple[str, str]:
    raw = json.dumps(authority, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), signed_payload_sha256(authority)


def _preview(
    graph: _ArchiveGraph, path: Path, file_hash: str, payload_hash: str
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_adapter=_ADAPTER,
        authority_file_sha256=file_hash,
        authority_payload_sha256=payload_hash,
        mutation_kind=_MUTATION,
        guard_set=_GUARDS,
        reason="recover the signed archived identity",
        gc=graph,
    )


def _apply(
    graph: _ArchiveGraph,
    path: Path,
    file_hash: str,
    payload_hash: str,
    manifest_hash: str,
) -> dict[str, Any]:
    return apply_signed_manifest(
        path,
        authority_adapter=_ADAPTER,
        authority_file_sha256=file_hash,
        authority_payload_sha256=payload_hash,
        mutation_kind=_MUTATION,
        guard_set=_GUARDS,
        reason="recover the signed archived identity",
        apply=True,
        manifest_sha256=manifest_hash,
        gc=graph,
    )


def test_signed_archive_reconstruction_creates_identity_and_allowlisted_edges(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())

    preview = _preview(graph, path, file_hash, payload_hash)
    applied = _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])

    assert preview["outcome"] == "would_apply"
    assert applied["outcome"] == "applied"
    assert graph.nodes["archived_temperature"]["properties"]["origin"] == "pipeline"
    assert graph.edges == [
        ("archived_temperature", "HAS_UNIT", "unit:eV", {"source": "archive"})
    ]


def test_duplicate_edge_rows_follow_merge_identity_and_refuse_false_count(
    tmp_path: Path,
) -> None:
    """Relationship properties do not create a second edge with the same key."""
    edges = [
        {
            "owner_id": "archived_temperature",
            "relationship_type": "HAS_UNIT",
            "direction": "outgoing",
            "counterpart_id": "unit:eV",
            "properties": {"source": source},
        }
        for source in ("first", "second")
    ]
    path = tmp_path / "duplicate-edge-authority.json"
    file_hash, payload_hash = _write_authority(path, _authority(edges=edges))
    graph = _ArchiveGraph()
    preview = _preview(graph, path, file_hash, payload_hash)

    with pytest.raises(SignedManifestConflict, match="counts differ"):
        _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])


def test_signed_archive_reconstruction_creates_allowlisted_counterparts(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    counterparts = [
        {
            "label": "StandardNameReview",
            "id": "review:archived-temperature",
            "properties": {
                "id": "review:archived-temperature",
                "standard_name_id": "archived_temperature",
                "review_axis": "name",
            },
        },
        {
            "label": "DocsRevision",
            "id": "docs:archived-temperature",
            "properties": {
                "id": "docs:archived-temperature",
                "standard_name_id": "archived_temperature",
                "documentation": "Archived temperature documentation.",
            },
        },
        {
            "label": "StandardNameSource",
            "id": "dd:archive/temperature",
            "properties": {
                "id": "dd:archive/temperature",
                "source_type": "dd",
                "source_id": "archive/temperature",
            },
        },
    ]
    edges = [
        {
            "owner_id": "archived_temperature",
            "relationship_type": "HAS_REVIEW",
            "direction": "outgoing",
            "counterpart_id": "review:archived-temperature",
            "properties": {},
        },
        {
            "owner_id": "archived_temperature",
            "relationship_type": "DOCS_REVISION_OF",
            "direction": "outgoing",
            "counterpart_id": "docs:archived-temperature",
            "properties": {},
        },
        {
            "owner_id": "archived_temperature",
            "relationship_type": "PRODUCED_NAME",
            "direction": "incoming",
            "counterpart_id": "dd:archive/temperature",
            "properties": {},
        },
    ]
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path, _authority(edges=edges, counterparts=counterparts)
    )

    preview = _preview(graph, path, file_hash, payload_hash)
    applied = _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])

    assert preview["outcome"] == "would_apply"
    assert applied["outcome"] == "applied"
    assert applied["counts"]["counterpart_rows"] == 3
    assert {label for label, _ in graph.counterparts} == {
        "DocsRevision",
        "StandardNameReview",
        "StandardNameSource",
        "Unit",
    }
    assert sorted(graph.edges, key=repr) == sorted(
        [
            ("archived_temperature", "HAS_REVIEW", "review:archived-temperature", {}),
            (
                "archived_temperature",
                "DOCS_REVISION_OF",
                "docs:archived-temperature",
                {},
            ),
            ("dd:archive/temperature", "PRODUCED_NAME", "archived_temperature", {}),
        ],
        key=repr,
    )


def test_signed_archive_reconstruction_uses_numeric_counterpart_identity(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    graph.counterparts[("StandardNameReview", 42)] = {
        "properties": {
            "id": 42,
            "standard_name_id": "archived_temperature",
            "review_axis": "name",
        }
    }
    edge = {
        "owner_id": "archived_temperature",
        "relationship_type": "HAS_REVIEW",
        "direction": "outgoing",
        "counterpart_id": 42,
        "properties": {"source": "archive"},
    }
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority(edges=[edge]))

    preview = _preview(graph, path, file_hash, payload_hash)
    applied = _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])

    assert preview["outcome"] == "would_apply"
    assert preview["manifest"]["state"]["counterparts"] == {
        "StandardNameReview:int:42": graph.counterparts[("StandardNameReview", 42)][
            "properties"
        ]
    }
    assert applied["outcome"] == "applied"
    assert graph.edges == [
        ("archived_temperature", "HAS_REVIEW", 42, {"source": "archive"})
    ]


def test_signed_archive_reconstruction_uses_identical_live_counterpart_without_creation(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    graph.counterparts[("StandardNameReview", "review:archived-temperature")] = {
        "properties": {
            "id": "review:archived-temperature",
            "standard_name_id": "archived_temperature",
            "review_axis": "name",
        }
    }
    edge = {
        "owner_id": "archived_temperature",
        "relationship_type": "HAS_REVIEW",
        "direction": "outgoing",
        "counterpart_id": "review:archived-temperature",
        "properties": {},
    }
    path = tmp_path / "authority.json"
    authority = _authority(edges=[edge])
    assert "counterparts" not in authority
    file_hash, payload_hash = _write_authority(path, authority)

    preview = _preview(graph, path, file_hash, payload_hash)
    applied = _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])

    assert preview["outcome"] == "would_apply"
    assert applied["outcome"] == "applied"
    assert applied["counts"]["counterpart_rows"] == 0
    assert applied["mutations"] == 2
    assert graph.edges == [
        ("archived_temperature", "HAS_REVIEW", "review:archived-temperature", {})
    ]


def test_archive_reconstruction_refuses_nonidentical_existing_counterpart(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    counterpart = {
        "label": "StandardNameReview",
        "id": "review:archived-temperature",
        "properties": {
            "id": "review:archived-temperature",
            "standard_name_id": "archived_temperature",
            "review_axis": "name",
        },
    }
    graph.counterparts[("StandardNameReview", "review:archived-temperature")] = {
        "properties": {
            **counterpart["properties"],
            "review_axis": "docs",
        }
    }
    edge = {
        "owner_id": "archived_temperature",
        "relationship_type": "HAS_REVIEW",
        "direction": "outgoing",
        "counterpart_id": "review:archived-temperature",
        "properties": {},
    }
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path, _authority(edges=[edge], counterparts=[counterpart])
    )

    preview = _preview(graph, path, file_hash, payload_hash)

    assert preview["outcome"] == "refused"
    assert "existing counterpart differs" in preview["refusals"][0]["reason"]


def test_archive_reconstruction_refuses_relationship_outside_registry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path,
        _authority(
            [
                {
                    "owner_id": "archived_temperature",
                    "relationship_type": "ARBITRARY_EDGE",
                    "direction": "outgoing",
                    "counterpart_id": "unit:eV",
                    "properties": {},
                }
            ]
        ),
    )

    with pytest.raises(
        SignedManifestAuthorityError, match="outside the reconstruction registry"
    ):
        _preview(_ArchiveGraph(), path, file_hash, payload_hash)


def test_archive_reconstruction_refuses_counterpart_label_outside_registry(
    tmp_path: Path,
) -> None:
    counterpart = {
        "label": "StandardNameChange",
        "id": "change:archived-temperature",
        "properties": {"id": "change:archived-temperature"},
    }
    edge = {
        "owner_id": "archived_temperature",
        "relationship_type": "HAS_INTERNAL_CHANGE",
        "direction": "outgoing",
        "counterpart_id": "change:archived-temperature",
        "properties": {},
    }
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path, _authority(edges=[edge], counterparts=[counterpart])
    )

    with pytest.raises(
        SignedManifestAuthorityError, match="counterpart label is outside"
    ):
        _preview(_ArchiveGraph(), path, file_hash, payload_hash)


def test_archive_reconstruction_refuses_missing_counterpart(tmp_path: Path) -> None:
    graph = _ArchiveGraph()
    graph.counterparts.clear()
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())

    preview = _preview(graph, path, file_hash, payload_hash)

    assert preview["outcome"] == "refused"
    assert "neither live nor included" in preview["refusals"][0]["reason"]


def test_archive_reconstruction_refuses_nonidentical_existing_node(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    graph.nodes["archived_temperature"] = {
        "properties": {
            "id": "archived_temperature",
            "origin": "pipeline",
            "status": "active",
        }
    }
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())

    preview = _preview(graph, path, file_hash, payload_hash)

    assert preview["outcome"] == "refused"
    assert (
        "differs from signed reconstruction state" in preview["refusals"][0]["reason"]
    )


def test_archive_reconstruction_refuses_archive_live_edge_count_mismatch(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    graph.extra_counts[("archived_temperature", "HAS_UNIT")] = 1
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(path, _authority())
    preview = _preview(graph, path, file_hash, payload_hash)

    with pytest.raises(SignedManifestConflict, match="relationship counts differ"):
        _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])


def test_archive_reconstruction_receipt_records_reinstated_roles_per_identity(
    tmp_path: Path,
) -> None:
    graph = _ArchiveGraph()
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path, _authority(archive_roles={"archived_temperature": {"HAS_UNIT": 1}})
    )

    preview = _preview(graph, path, file_hash, payload_hash)
    applied = _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])

    assert preview["identity_roles"] == {
        "archived_temperature": {"reinstated": {}, "unreinstatable": {}}
    }
    assert applied["identity_roles"] == {
        "archived_temperature": {"reinstated": {"HAS_UNIT": 1}, "unreinstatable": {}}
    }


def test_archive_reconstruction_receipt_names_unreinstatable_archived_roles(
    tmp_path: Path,
) -> None:
    """A role with no reconstruction route is reported, not silently dropped."""
    graph = _ArchiveGraph()
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path,
        _authority(
            archive_roles={"archived_temperature": {"HAS_UNIT": 1, "EVIDENCED_BY": 2}}
        ),
    )

    preview = _preview(graph, path, file_hash, payload_hash)
    applied = _apply(graph, path, file_hash, payload_hash, preview["manifest_sha256"])

    roles = applied["identity_roles"]["archived_temperature"]
    assert roles["unreinstatable"] == {"EVIDENCED_BY": 2}
    assert roles["reinstated"] == {"HAS_UNIT": 1}
    assert preview["identity_roles"]["archived_temperature"]["unreinstatable"] == {
        "EVIDENCED_BY": 2
    }
    assert all(
        relationship_type != "EVIDENCED_BY"
        for _, relationship_type, _, _ in graph.edges
    )


def test_archive_reconstruction_refuses_archived_role_beyond_reconstruction_edges(
    tmp_path: Path,
) -> None:
    path = tmp_path / "authority.json"
    file_hash, payload_hash = _write_authority(
        path, _authority(archive_roles={"archived_temperature": {"HAS_UNIT": 2}})
    )

    with pytest.raises(
        SignedManifestAuthorityError, match="cannot reinstate an archived role"
    ):
        _preview(_ArchiveGraph(), path, file_hash, payload_hash)
