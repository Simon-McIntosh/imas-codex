"""Disposable-graph contract for signed unbound ordinary-source attachment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.signed_manifest import (
    SignedManifestConflict,
    apply_signed_manifest,
    signed_payload_sha256,
)

_PATH = "spectrometer_visible/channel/detector/centre/phi"
_SOURCE_ID = f"dd:{_PATH}"
_TARGET_ID = "toroidal_coordinate_of_detector"
_OPERATION = "attach_unbound_standard_name_source"
_REASON = "attach the exact independently adjudicated unbound source"
_SELECTION = {
    "id": "artifact-rows",
    "mode": "exact_complete_signed_cohort",
    "predicate": "artifact-rows",
}


def _row() -> dict[str, Any]:
    return {
        "id": _SOURCE_ID,
        "identity": {
            "id": _SOURCE_ID,
            "kind": "source",
            "source_id": _SOURCE_ID,
            "target_id": _TARGET_ID,
        },
        "participants": [
            {
                "id": _SOURCE_ID,
                "kind": "node",
                "graph_label": "StandardNameSource",
            },
            {
                "id": _TARGET_ID,
                "kind": "node",
                "graph_label": "StandardName",
            },
        ],
        "selection": _SELECTION,
        "mutations": [
            {
                "id": "attach-authoritative-target",
                "order": 1,
                "kind": "add_relationship",
                "participant_id": _TARGET_ID,
                "arguments": {
                    "relationship_type": "PRODUCED_NAME",
                    "start_id": _SOURCE_ID,
                    "end_id": _TARGET_ID,
                },
            },
            {
                "id": "advance-source-lifecycle",
                "order": 2,
                "kind": "set_properties",
                "participant_id": _SOURCE_ID,
                "arguments": {
                    "properties": {
                        "status": "attached",
                        "produced_sn_id": _TARGET_ID,
                        "claimed_at": None,
                        "claim_token": None,
                        "last_error": None,
                    }
                },
            },
        ],
        "guards": [
            {
                "id": "out-of-allowlist-immutability",
                "kind": "collateral_immutability",
                "implementation": "out-of-allowlist-immutability",
                "participant_ids": [],
            }
        ],
        "orphan_policy": "refuse",
    }


def _write_authority(path: Path) -> tuple[str, str]:
    authority: dict[str, Any] = {
        "schema": "imas-codex.repair-authority.v1",
        "operation_id": "unbound-ordinary-source-attachment",
        "authority_mode": "external_reviewed",
        "rows": [_row()],
        "repair_rows": [_SOURCE_ID],
        "selection": _SELECTION,
        "receipt_policy": {
            "id": "one-per-unbound-source-attachment",
            "operation": _OPERATION,
            "cardinality": "per_target",
            "expected_count": "admitted_rows",
            "link_participant_kind": "source",
            "replay_projection": ["manifest_sha256", "row_id"],
        },
        "orphan_policy": "refuse",
    }
    payload_sha256 = signed_payload_sha256(authority)
    authority["signature"] = {
        "canonicalization": "json-sort-keys-v1",
        "sha256": payload_sha256,
    }
    raw = json.dumps(authority, sort_keys=True, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest(), payload_sha256


@pytest.fixture
def client(disposable_neo4j: tuple[str, str]) -> Iterator[GraphClient]:
    uri, password = disposable_neo4j
    graph = GraphClient(
        uri=uri,
        username="neo4j",
        password=password,
        graph_name="unbound-source-attachment",
    )
    graph.query("MATCH (node) DETACH DELETE node")
    yield graph
    graph.query("MATCH (node) DETACH DELETE node")
    graph.close()


def _seed(client: GraphClient, *, dd_unit: str = "rad") -> None:
    client.query(
        """
        CREATE (target:StandardName {
          id: $target_id, name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', source_paths: ['dd:retained/path'], unit: 'rad'
        })
        CREATE (dd:IMASNode {id: $path, unit: $dd_unit})
        CREATE (source:StandardNameSource {
          id: $source_id, source_type: 'dd', source_id: $path,
          status: 'extracted', produced_sn_id: null, last_error: 'prior miss'
        })
        CREATE (source)-[:FROM_DD_PATH]->(dd)
        CREATE (:StandardName {
          id: 'collateral_name', name_stage: 'accepted',
          validation_status: 'valid', status: 'draft', source_paths: []
        })
        CREATE (:StandardNameSource {
          id: 'dd:collateral/path', source_type: 'dd',
          source_id: 'collateral/path', status: 'stale'
        })
        """,
        target_id=_TARGET_ID,
        path=_PATH,
        source_id=_SOURCE_ID,
        dd_unit=dd_unit,
    )


def _preview(
    client: GraphClient, tmp_path: Path
) -> tuple[Path, str, str, dict[str, Any]]:
    authority = tmp_path / "authority.json"
    file_digest, payload_digest = _write_authority(authority)
    preview = apply_signed_manifest(
        authority,
        authority_file_sha256=file_digest,
        authority_payload_sha256=payload_digest,
        reason=_REASON,
        gc=client,
    )
    return authority, file_digest, payload_digest, preview


def _apply(
    client: GraphClient,
    authority: Path,
    file_digest: str,
    payload_digest: str,
    manifest_digest: str,
) -> dict[str, Any]:
    return apply_signed_manifest(
        authority,
        authority_file_sha256=file_digest,
        authority_payload_sha256=payload_digest,
        reason=_REASON,
        apply=True,
        manifest_sha256=manifest_digest,
        gc=client,
    )


def _happy_apply(
    client: GraphClient, tmp_path: Path
) -> tuple[dict[str, Any], tuple[Any, ...]]:
    _seed(client)
    authority, file_digest, payload_digest, preview = _preview(client, tmp_path)
    applied = _apply(
        client,
        authority,
        file_digest,
        payload_digest,
        preview["manifest_sha256"],
    )
    return applied, (authority, file_digest, payload_digest, preview)


@pytest.mark.graph
def test_preview_reports_would_apply(client: GraphClient, tmp_path: Path) -> None:
    _seed(client)

    _, _, _, preview = _preview(client, tmp_path)

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {"authority_rows": 1, "admitted": 1, "refused": 0}


@pytest.mark.graph
def test_apply_changed_equals_admitted_rows(
    client: GraphClient, tmp_path: Path
) -> None:
    applied, _ = _happy_apply(client, tmp_path)

    assert applied["outcome"] == "applied"
    assert applied["changed"] == applied["counts"]["admitted"] == 1
    state = client.query(
        """
        MATCH (source:StandardNameSource {id: $source_id})
              -[:PRODUCED_NAME]->(target:StandardName {id: $target_id})
        MATCH (source)-[:FROM_DD_PATH]->(dd:IMASNode)
              -[:HAS_STANDARD_NAME]->(target)
        RETURN source.status AS status, source.produced_sn_id AS scalar,
               target.source_paths AS source_paths
        """,
        source_id=_SOURCE_ID,
        target_id=_TARGET_ID,
    )[0]
    assert state == {
        "status": "attached",
        "scalar": _TARGET_ID,
        "source_paths": ["dd:retained/path", _SOURCE_ID],
    }


@pytest.mark.graph
def test_receipt_rows_equal_admitted_rows(client: GraphClient, tmp_path: Path) -> None:
    applied, _ = _happy_apply(client, tmp_path)

    assert applied["receipt_rows"] == applied["counts"]["admitted"] == 1
    assert client.query(
        "MATCH (change:StandardNameChange {operation: $operation}) RETURN count(change) AS count",
        operation=_OPERATION,
    ) == [{"count": 1}]


@pytest.mark.graph
def test_replay_is_write_free(client: GraphClient, tmp_path: Path) -> None:
    applied, authority_parts = _happy_apply(client, tmp_path)
    authority, file_digest, payload_digest, preview = authority_parts
    before = client.query(
        "MATCH (node) OPTIONAL MATCH (node)-[edge]->() RETURN count(node) AS nodes, count(edge) AS edges"
    )[0]

    replay = _apply(
        client,
        authority,
        file_digest,
        payload_digest,
        preview["manifest_sha256"],
    )
    after = client.query(
        "MATCH (node) OPTIONAL MATCH (node)-[edge]->() RETURN count(node) AS nodes, count(edge) AS edges"
    )[0]

    assert applied["outcome"] == "applied"
    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert replay["persistent_writes"] == 0
    assert after == before


@pytest.mark.graph
def test_already_bound_source_refusal_is_exact(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    client.query(
        """
        CREATE (other:StandardName {
          id: 'other_name', name_stage: 'accepted', validation_status: 'valid',
          status: 'draft', source_paths: [$source_id]
        })
        WITH other
        MATCH (source:StandardNameSource {id: $source_id}), (dd:IMASNode {id: $path})
        CREATE (source)-[:PRODUCED_NAME]->(other)
        CREATE (dd)-[:HAS_STANDARD_NAME]->(other)
        SET source.status = 'attached', source.produced_sn_id = other.id
        """,
        source_id=_SOURCE_ID,
        path=_PATH,
    )

    _, _, _, preview = _preview(client, tmp_path)

    assert preview["refusals"] == [
        {"row_id": _SOURCE_ID, "reason": "ordinary source is already bound"}
    ]


@pytest.mark.graph
def test_compare_and_set_drift_refuses_apply(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    authority, file_digest, payload_digest, preview = _preview(client, tmp_path)
    client.query(
        "MATCH (source:StandardNameSource {id: $source_id}) SET source.last_error = 'drifted'",
        source_id=_SOURCE_ID,
    )

    with pytest.raises(
        SignedManifestConflict,
        match="^fresh signed-manifest closure does not match authorized SHA-256$",
    ):
        _apply(
            client,
            authority,
            file_digest,
            payload_digest,
            preview["manifest_sha256"],
        )


@pytest.mark.graph
def test_unit_disagreeing_pairing_refusal_is_exact(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client, dd_unit="m")

    _, _, _, preview = _preview(client, tmp_path)

    assert preview["refusals"] == [
        {
            "row_id": _SOURCE_ID,
            "reason": (
                f"source attachment rejected: {_SOURCE_ID}: unit dimensionality "
                f"mismatch: path '{_PATH}' declares 'm' but SN '{_TARGET_ID}' "
                "declares 'rad' — physically distinct quantities"
            ),
        }
    ]


@pytest.mark.graph
def test_out_of_allowlist_rows_are_immutable(
    client: GraphClient, tmp_path: Path
) -> None:
    _seed(client)
    before = client.query(
        """
        MATCH (name:StandardName {id: 'collateral_name'}),
              (source:StandardNameSource {id: 'dd:collateral/path'})
        RETURN properties(name) AS name, properties(source) AS source
        """
    )[0]

    authority, file_digest, payload_digest, preview = _preview(client, tmp_path)
    _apply(
        client,
        authority,
        file_digest,
        payload_digest,
        preview["manifest_sha256"],
    )
    after = client.query(
        """
        MATCH (name:StandardName {id: 'collateral_name'}),
              (source:StandardNameSource {id: 'dd:collateral/path'})
        RETURN properties(name) AS name, properties(source) AS source
        """
    )[0]

    assert after == before


# ---------------------------------------------------------------------------
# Mocked — the ``sn attach`` verb: the symmetric counterpart of ``sn detach``
# ---------------------------------------------------------------------------

_BETA_PATH = "core_profiles/global_quantities/beta_tor"
_BETA_SOURCE = f"dd:{_BETA_PATH}"
_BETA_NAME = "toroidal_beta"
_SIBLING_PATH = "equilibrium/time_slice/global_quantities/beta_tor"
_ATTACH_REASON = "the toroidal beta of the plasma, already named"


class _AttachGraph:
    """In-memory stand-in answering exactly the queries attach and detach issue.

    Keyed on the module's own query CONSTANTS rather than on query text, so a
    rewritten statement surfaces as an unrecognized query instead of silently
    passing. ``test_attach_and_detach_name_the_same_edges_and_scalars`` reads the
    Cypher itself, which is what keeps this stand-in from drifting away from it.
    """

    def __init__(
        self,
        *,
        dd_unit: str = "1",
        sn_unit: str = "1",
        name_stage: str = "accepted",
        target_exists: bool = True,
        dd_exists: bool = True,
        source_exists: bool = True,
        bound_to: str | None = None,
    ) -> None:
        self.dd_paths: dict[str, dict[str, Any]] = {}
        if dd_exists:
            self.dd_paths[_BETA_PATH] = {"unit": dd_unit, "documentation": ""}
        self.dd_paths[_SIBLING_PATH] = {"unit": sn_unit, "documentation": ""}
        self.sources: dict[str, dict[str, Any]] = {}
        if source_exists:
            # The starting shape of a source that was extracted and never
            # composed: every lifecycle scalar explicitly empty, so a
            # round-trip comparison is exact rather than key-presence noise.
            self.sources[_BETA_SOURCE] = {
                "id": _BETA_SOURCE,
                "source_type": "dd",
                "dd_path": _BETA_PATH,
                "status": "extracted",
                "produced_sn_id": None,
                "composed_at": None,
                "claimed_at": None,
                "claim_token": None,
                "attempt_count": 0,
                "last_error": None,
            }
        self.sources[f"dd:{_SIBLING_PATH}"] = {
            "id": f"dd:{_SIBLING_PATH}",
            "source_type": "dd",
            "dd_path": _SIBLING_PATH,
            "status": "attached",
            "produced_sn_id": _BETA_NAME if target_exists else None,
            "composed_at": "seeded",
            "claimed_at": None,
            "claim_token": None,
            "attempt_count": 0,
            "last_error": None,
        }
        self.names: dict[str, dict[str, Any]] = {}
        if target_exists:
            self.names[_BETA_NAME] = {
                "id": _BETA_NAME,
                "name_stage": name_stage,
                "unit": sn_unit,
                "source_paths": [f"dd:{_SIBLING_PATH}"],
                "origin": "dd",
            }
        self.names["electron_temperature"] = {
            "id": "electron_temperature",
            "name_stage": "accepted",
            "unit": "eV",
            "source_paths": [],
            "origin": "dd",
        }
        self.produced: set[tuple[str, str]] = set()
        self.projected: set[tuple[str, str]] = set()
        if target_exists:
            self.produced.add((f"dd:{_SIBLING_PATH}", _BETA_NAME))
            self.projected.add((_SIBLING_PATH, _BETA_NAME))
        if bound_to:
            self.names.setdefault(
                bound_to,
                {
                    "id": bound_to,
                    "name_stage": "accepted",
                    "unit": sn_unit,
                    "source_paths": [_BETA_SOURCE],
                    "origin": "dd",
                },
            )
            self.produced.add((_BETA_SOURCE, bound_to))
            self.projected.add((_BETA_PATH, bound_to))
            self.sources[_BETA_SOURCE]["status"] = "attached"
            self.sources[_BETA_SOURCE]["produced_sn_id"] = bound_to
        self.changes: list[dict[str, Any]] = []
        self.writes: list[str] = []

    # -- shape ------------------------------------------------------------
    def shape(self) -> dict[str, Any]:
        """Everything a realization asserts, excluding pure mtimes.

        ``updated_at`` is deliberately out: it records WHEN the name last
        changed, not what the graph says, so requiring it to be restored would
        demand a detach that lies about its own moment.
        """
        return {
            "sources": deepcopy(self.sources),
            "names": {
                sn_id: {k: v for k, v in props.items() if k != "updated_at"}
                for sn_id, props in self.names.items()
            },
            "produced": sorted(self.produced),
            "projected": sorted(self.projected),
        }

    # -- query dispatch ---------------------------------------------------
    def query(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        from imas_codex.standard_names import attachment_audit as mod

        if statement == mod._ATTACH_PREFLIGHT_QUERY:
            return self._attach_preflight(**params)
        if statement == mod._PAIRING_GUARD_QUERY:
            return self._pairing_guard(**params)
        if statement == mod._ATTACH_QUERY:
            self.writes.append("attach")
            return self._attach(**params)
        if statement == mod._DETACH_QUERY:
            self.writes.append("detach")
            return self._detach(**params)
        if statement == mod._DETACH_PROJECTION_QUERY:
            self.writes.append("detach_projection")
            raise AssertionError(
                "the round trip has provenance; no dangling projection"
            )
        if "RETURN src.id AS source_node_id" in statement:
            return self._detach_preflight(**params)
        if "CREATE (change:StandardNameChange" in statement:
            self.changes.append(dict(params))
            return []
        raise AssertionError(f"unrecognized query: {statement[:120]!r}")

    def close(self) -> None:  # pragma: no cover - the caller owns this handle
        raise AssertionError("attach_one_source must not close a borrowed handle")

    # -- readers ----------------------------------------------------------
    def _live_names(self, source_id: str, historical: list[str]) -> list[str]:
        return sorted(
            {
                sn_id
                for src, sn_id in self.produced
                if src == source_id
                and (self.names.get(sn_id, {}).get("name_stage") or "")
                not in historical
            }
        )

    def _attach_preflight(
        self, *, dd_path: str, sn_id: str, historical: list[str]
    ) -> list[dict[str, Any]]:
        if dd_path not in self.dd_paths:
            return []
        source = next(
            (s for s in self.sources.values() if s["dd_path"] == dd_path), None
        )
        name = self.names.get(sn_id)
        return [
            {
                "source_node_id": source["id"] if source else None,
                "source_status": source["status"] if source else None,
                "live_names": (
                    self._live_names(source["id"], historical) if source else []
                ),
                "name_exists": name is not None,
                "name_stage": name["name_stage"] if name else None,
                "projected": (dd_path, sn_id) in self.projected,
            }
        ]

    def _pairing_guard(
        self, *, sn_id: str, source_ids: list[str]
    ) -> list[dict[str, Any]]:
        name = self.names.get(sn_id, {})
        existing = sorted(
            {
                self.sources[src]["dd_path"]
                for src, bound in self.produced
                if bound == sn_id and src in self.sources
            }
        )
        rows = []
        for source_id in source_ids:
            source = self.sources.get(source_id)
            if source is None:
                rows.append({"source_id": source_id, "source_type": None})
                continue
            dd = self.dd_paths.get(source["dd_path"], {})
            rows.append(
                {
                    "source_id": source_id,
                    "source_type": source["source_type"],
                    "dd_path": source["dd_path"],
                    "dd_unit": dd.get("unit"),
                    "sn_unit": name.get("unit"),
                    "already_bound": (source_id, sn_id) in self.produced,
                    "existing_dd_paths": existing,
                    "name_stage": name.get("name_stage"),
                    "dd_documentation": dd.get("documentation"),
                }
            )
        return rows

    def _detach_preflight(
        self, *, dd_path: str, sn_id: str, historical: list[str]
    ) -> list[dict[str, Any]]:
        source = next(
            (
                s
                for s in self.sources.values()
                if s["dd_path"] == dd_path and (s["id"], sn_id) in self.produced
            ),
            None,
        )
        name = self.names.get(sn_id, {})
        others = 0
        if source:
            others = len(
                [n for n in self._live_names(source["id"], historical) if n != sn_id]
            )
        return [
            {
                "source_node_id": source["id"] if source else None,
                "other_live_names": others,
                "projected": (dd_path, sn_id) in self.projected,
                "name_attachments": len(
                    [1 for _, bound in self.projected if bound == sn_id]
                ),
                "structural_parent": name.get("origin") == "derived",
            }
        ]

    # -- writers ----------------------------------------------------------
    def _attach(
        self, *, source_node_id: str, dd_path: str, sn_id: str
    ) -> list[dict[str, Any]]:
        source = self.sources[source_node_id]
        self.produced.add((source_node_id, sn_id))
        self.projected.add((dd_path, sn_id))
        source.update(
            status="attached",
            composed_at="written",
            claimed_at=None,
            claim_token=None,
            produced_sn_id=sn_id,
            last_error=None,
        )
        name = self.names[sn_id]
        uri = f"dd:{dd_path}"
        if uri not in name["source_paths"] and dd_path not in name["source_paths"]:
            name["source_paths"] = [*name["source_paths"], uri]
        name["updated_at"] = "written"
        return [{"attached": 1}]

    def _detach(
        self, *, items: list[dict[str, Any]], historical: list[str]
    ) -> list[dict[str, Any]]:
        for item in items:
            source_id = item["source_node_id"]
            sn_id = item["sn_id"]
            dd_path = item["dd_path"]
            self.produced.discard((source_id, sn_id))
            self.projected.discard((dd_path, sn_id))
            name = self.names[sn_id]
            name["source_paths"] = [
                p for p in name["source_paths"] if p not in (f"dd:{dd_path}", dd_path)
            ]
            remaining = self._live_names(source_id, historical)
            source = self.sources[source_id]
            source["produced_sn_id"] = remaining[0] if remaining else None
            if item["reroute"]:
                source.update(
                    status="extracted",
                    composed_at=None,
                    attempt_count=0,
                    claimed_at=None,
                    claim_token=None,
                )
        return [{"detached": len(items)}]


def test_attach_binds_an_unbound_source_to_the_name_it_realizes() -> None:
    """The motivating case: an extracted, unbound beta path onto an accepted name."""
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph()
    assert graph.sources[_BETA_SOURCE]["status"] == "extracted"

    result = attach_one_source(_BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph)

    assert result["ok"] is True
    assert result["source_node_id"] == _BETA_SOURCE
    assert result["name_stage"] == "accepted"
    # Every assertion a realization makes, and the lifecycle move behind it.
    assert (_BETA_SOURCE, _BETA_NAME) in graph.produced
    assert (_BETA_PATH, _BETA_NAME) in graph.projected
    assert graph.names[_BETA_NAME]["source_paths"] == [
        f"dd:{_SIBLING_PATH}",
        _BETA_SOURCE,
    ]
    assert graph.sources[_BETA_SOURCE]["status"] == "attached"
    assert graph.sources[_BETA_SOURCE]["produced_sn_id"] == _BETA_NAME
    assert graph.sources[_BETA_SOURCE]["composed_at"] == "written"
    # The judgement and its reason survive in the ledger, as a detach's does.
    assert len(graph.changes) == 1
    change = graph.changes[0]
    assert change["operation"] == "attach_unbound_standard_name_source"
    assert change["from_name"] == _BETA_PATH
    assert change["to_name"] == _BETA_NAME
    assert _ATTACH_REASON in change["reason"]


def test_attach_then_detach_returns_the_graph_to_its_starting_shape() -> None:
    """The verbs are inverse: neither leaves a residue the other cannot remove."""
    from imas_codex.standard_names.attachment_audit import (
        attach_one_source,
        detach_one_attachment,
    )

    graph = _AttachGraph()
    before = graph.shape()

    assert attach_one_source(_BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph)[
        "ok"
    ]
    assert graph.shape() != before, "the attach wrote nothing"

    assert detach_one_attachment(
        _BETA_PATH, _BETA_NAME, reason="reverting the judgement", gc=graph
    )["ok"]

    assert graph.shape() == before
    assert graph.writes == ["attach", "detach"]
    # Both judgements stay in the ledger — a restored shape is not erased history.
    assert [c["operation"] for c in graph.changes] == [
        "attach_unbound_standard_name_source",
        "detach_inconsistent_attachment",
    ]


def test_attach_and_detach_name_the_same_edges_and_scalars() -> None:
    """Guards the stand-in above against drifting away from the real Cypher."""
    from imas_codex.standard_names import attachment_audit as mod

    for token in ("PRODUCED_NAME", "HAS_STANDARD_NAME", "source_paths"):
        assert token in mod._ATTACH_QUERY, token
        assert token in mod._DETACH_QUERY, token
    for scalar in ("status", "composed_at", "claimed_at", "claim_token"):
        assert f"src.{scalar}" in mod._ATTACH_QUERY, scalar
        assert f"src.{scalar}" in mod._DETACH_QUERY, scalar
    assert "src.produced_sn_id" in mod._ATTACH_QUERY
    assert "src.produced_sn_id" in mod._DETACH_QUERY


def test_attach_refuses_a_source_already_bound_to_a_live_name() -> None:
    """Re-pointing is a detach then an attach; one verb would hide the middle."""
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph(bound_to="normalized_toroidal_beta")
    before = graph.shape()

    result = attach_one_source(_BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph)

    assert result["ok"] is False
    assert "normalized_toroidal_beta" in result["reason"]
    assert "sn detach first" in result["reason"]
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


def test_attach_refuses_a_pairing_the_mechanical_guard_rejects() -> None:
    """An attach that bypassed the guard would manufacture the audit's own work."""
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph()
    before = graph.shape()

    result = attach_one_source(
        _BETA_PATH,
        "electron_temperature",
        reason="dimensionless beta is not a temperature",
        gc=graph,
    )

    assert result["ok"] is False
    assert "the consistency guard rejects this pairing" in result["reason"]
    assert "unit dimensionality mismatch" in result["reason"]
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


def test_attach_refuses_a_target_name_that_does_not_exist() -> None:
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph(target_exists=False)
    before = graph.shape()

    result = attach_one_source(_BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph)

    assert result["ok"] is False
    assert result["reason"] == f"{_BETA_NAME!r} does not exist"
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


@pytest.mark.parametrize("name_stage", ["superseded", "exhausted", "contested"])
def test_attach_refuses_a_terminal_target(name_stage: str) -> None:
    """Binding onto a terminal identity is the corruption the recovery repairs."""
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph(name_stage=name_stage)
    before = graph.shape()

    result = attach_one_source(_BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph)

    assert result["ok"] is False
    assert f"name_stage {name_stage!r}" in result["reason"]
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


def test_attach_refuses_a_dd_path_the_graph_does_not_carry() -> None:
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph(dd_exists=False)
    before = graph.shape()

    result = attach_one_source(_BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph)

    assert result["ok"] is False
    assert result["reason"] == f"{_BETA_PATH!r} is not a DD path in the graph"
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


def test_attach_refuses_a_dd_path_with_no_source_node() -> None:
    """A path nobody extracted has nothing to bind — extraction comes first."""
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph(source_exists=False)
    before = graph.shape()

    result = attach_one_source(_BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph)

    assert result["ok"] is False
    assert "no StandardNameSource node" in result["reason"]
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


def test_attach_dry_run_does_not_write() -> None:
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph()
    before = graph.shape()

    result = attach_one_source(
        _BETA_PATH, _BETA_NAME, reason=_ATTACH_REASON, gc=graph, dry_run=True
    )

    assert result["ok"] is True and result["dry_run"] is True
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


def test_attach_command_reports_the_binding_it_made() -> None:
    from click.testing import CliRunner

    from imas_codex.cli.sn import sn
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph()

    def _attach(dd_path: str, sn_id: str, **kwargs: Any) -> dict[str, Any]:
        return attach_one_source(dd_path, sn_id, gc=graph, **kwargs)

    with patch(
        "imas_codex.standard_names.attachment_audit.attach_one_source",
        side_effect=_attach,
    ):
        result = CliRunner().invoke(
            sn, ["attach", _BETA_PATH, _BETA_NAME, "--reason", _ATTACH_REASON]
        )

    assert result.exit_code == 0, result.output
    assert f"attached {_BETA_PATH} to {_BETA_NAME}" in result.output
    assert (_BETA_SOURCE, _BETA_NAME) in graph.produced


def test_attach_command_exits_non_zero_and_writes_nothing_on_a_refusal() -> None:
    """The refusal reason reaches the operator; the graph is untouched."""
    from click.testing import CliRunner

    from imas_codex.cli.sn import sn
    from imas_codex.standard_names.attachment_audit import attach_one_source

    graph = _AttachGraph(bound_to="normalized_toroidal_beta")
    before = graph.shape()

    def _attach(dd_path: str, sn_id: str, **kwargs: Any) -> dict[str, Any]:
        return attach_one_source(dd_path, sn_id, gc=graph, **kwargs)

    with patch(
        "imas_codex.standard_names.attachment_audit.attach_one_source",
        side_effect=_attach,
    ):
        result = CliRunner().invoke(
            sn, ["attach", _BETA_PATH, _BETA_NAME, "--reason", _ATTACH_REASON]
        )

    assert result.exit_code != 0
    assert "normalized_toroidal_beta" in result.output
    assert graph.writes == [] and graph.changes == []
    assert graph.shape() == before


def test_attach_requires_a_reason() -> None:
    """An attach is a physics judgement and must carry its argument."""
    from click.testing import CliRunner

    from imas_codex.cli.sn import sn

    result = CliRunner().invoke(sn, ["attach", _BETA_PATH, _BETA_NAME])

    assert result.exit_code != 0
    assert "--reason" in result.output
