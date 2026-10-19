"""Signed recovery-binding unbind removes a recorded producer from an exhausted name.

The program runs inside the signed-repair envelope.  It may remove a
``PRODUCED_NAME`` edge only when the edge exists, the source is live, and the
name's stage is ``exhausted``; the name, its stage, its other edges and the
``StandardNameSource`` node are otherwise left exactly as they were.  A graph
that cannot be reached is emulated here, so these tests exercise the program's
own decisions rather than a live database.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names import signed_manifest as operator
from imas_codex.standard_names.signed_manifest import (
    SignedManifestAuthorityError,
    apply_signed_manifest,
)
from imas_codex.standard_names.source_invariant import (
    MissingStandardNameSource,
    require_producing_sources,
)

_REASON = "unbind a recorded recovery binding from an exhausted name"
_GUARD_SET = ("exhausted-target-live-source", "out-of-allowlist-immutability")


class _Transaction:
    """Emulate the marker-tagged Cypher the unbind program issues."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        self.calls: list[str] = []
        self.committed = False
        self.rolled_back = False

    @property
    def closed(self) -> bool:
        return self.committed or self.rolled_back

    def _binding(self, source_id: str, name_id: str) -> dict[str, Any] | None:
        for binding in self.state["bindings"]:
            if binding["source_id"] == source_id and binding["name_id"] == name_id:
                return binding
        return None

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append(query)
        if "RECOVERY_UNBIND_CLOSURE" in query:
            rows = []
            for expected in params["rows"]:
                name_id = expected["name_id"]
                source_id = expected["source_id"]
                name = self.state["names"].get(name_id)
                source = self.state["sources"].get(source_id)
                binding = self._binding(source_id, name_id)
                rows.append(
                    {
                        "name_id": name_id,
                        "source_id": source_id,
                        "source_element_id": None if source is None
                        else source["element_id"],
                        "source_properties": deepcopy(source["properties"])
                        if source
                        else None,
                        "name_element_id": None if name is None
                        else name["element_id"],
                        "name_properties": deepcopy(name["properties"])
                        if name
                        else None,
                        "binding_element_id": None if binding is None
                        else binding["element_id"],
                        "binding_present": binding is not None,
                    }
                )
            rows.sort(key=lambda row: (row["name_id"], row["source_id"]))
            return rows
        if "RECOVERY_UNBIND_REPLAY" in query:
            recorded = set(self.state["changes"])
            return [
                {"event_id": event_id, "recorded": event_id in recorded}
                for event_id in sorted(params["event_ids"])
            ]
        if "RECOVERY_UNBIND_MUTATION" in query:
            binding = self._binding(params["source_id"], params["name_id"])
            if (
                binding is None
                or binding["element_id"] != params["binding_element_id"]
            ):
                return [{"changed": 0}]
            self.state["bindings"].remove(binding)
            return [{"changed": 1}]
        if "RECOVERY_UNBIND_RECEIPT" in query:
            written = []
            for row in params["rows"]:
                change = {
                    "id": row["change_id"],
                    "from_name": row["source_id"],
                    "to_name": row["name_id"],
                    "operation": params["operation"],
                    "manifest_sha256": params["manifest_sha256"],
                }
                self.state["changes"][row["change_id"]] = change
                written.append({"change_id": row["change_id"]})
            return sorted(written, key=lambda row: row["change_id"])
        if "RECOVERY_UNBIND_POSTCONDITION" in query:
            rows = []
            for expected in params["rows"]:
                name_id = expected["name_id"]
                source_id = expected["source_id"]
                name = self.state["names"][name_id]
                source = self.state["sources"][source_id]
                rows.append(
                    {
                        "name_id": name_id,
                        "source_id": source_id,
                        "bindings": 1
                        if self._binding(source_id, name_id) is not None
                        else 0,
                        "name_stage": name["properties"].get("name_stage"),
                        "source_status": source["properties"].get("status"),
                    }
                )
            rows.sort(key=lambda row: row["name_id"])
            return rows
        if "SOURCE_INVARIANT" in query:
            return self._source_invariant(params)
        raise AssertionError(f"unexpected query: {query}")

    def _source_invariant(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        exempt = set(params["exempt_ids"])
        rows = []
        for name_id in params["ids"]:
            name = self.state["names"].get(name_id)
            if name is None:
                continue
            properties = name["properties"]
            stage = properties.get("name_stage") or ""
            status = properties.get("status") or ""
            if params["retired_is_exempt"] and (
                stage == "superseded" or status == "superseded"
            ):
                continue
            if properties.get("origin") == "derived":
                continue
            producers = sum(
                1
                for binding in self.state["bindings"]
                if binding["name_id"] == name_id
                and self.state["sources"][binding["source_id"]]["properties"].get(
                    "status"
                )
                != "stale"
            )
            if producers:
                continue
            if stage == "exhausted" and name_id in exempt:
                continue
            rows.append({"id": name_id})
        return sorted(rows, key=lambda row: row["id"])

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


class _Session:
    def __init__(self, transaction: _Transaction) -> None:
        self.transaction = transaction

    def begin_transaction(self) -> _Transaction:
        return self.transaction

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class FakeGraph:
    """Minimal in-memory graph holding exactly the nodes unbind touches."""

    def __init__(self) -> None:
        self.names: dict[str, dict[str, Any]] = {}
        self.sources: dict[str, dict[str, Any]] = {}
        self.bindings: list[dict[str, Any]] = []
        self.changes: dict[str, dict[str, Any]] = {}
        self.transactions: list[_Transaction] = []

    def add_name(self, name_id: str, name_stage: str, **extra: Any) -> None:
        self.names[name_id] = {
            "element_id": f"sn:{name_id}",
            "properties": {"id": name_id, "name_stage": name_stage, **extra},
        }

    def add_source(self, source_id: str, status: str = "composed") -> None:
        self.sources[source_id] = {
            "element_id": f"src:{source_id}",
            "properties": {"id": source_id, "status": status},
        }

    def bind(self, source_id: str, name_id: str) -> None:
        self.bindings.append(
            {
                "element_id": f"rel:{source_id}->{name_id}",
                "source_id": source_id,
                "name_id": name_id,
            }
        )

    def session(self) -> _Session:
        transaction = _Transaction(self)
        self.transactions.append(transaction)
        return _Session(transaction)

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        """Answer a standalone read, as the source invariant's callers expect."""
        return _Transaction(self).run(cypher, **params)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def snapshot(self) -> dict[str, Any]:
        return {
            "names": deepcopy(self.names),
            "sources": deepcopy(self.sources),
            "bindings": deepcopy(self.bindings),
            "changes": deepcopy(self.changes),
        }


def _write_authority(path: Path, rows: list[dict[str, str]]) -> None:
    canonical = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ) + "\n"
    authority = {
        "schema": operator._RECOVERY_UNBIND_SCHEMA,
        "rows": rows,
        "signature": {
            "algorithm": "sha256",
            "canonicalization": "jq -cS '.rows'",
            "scope": "rows",
            "digest": hashlib.sha256(canonical.encode()).hexdigest(),
        },
    }
    path.write_text(json.dumps(authority, indent=2))


def _apply(
    graph: FakeGraph,
    path: Path,
    name_ids: list[str],
    **overrides: Any,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "authority_adapter": operator._RECOVERY_UNBIND_ADAPTER,
        "mutation_kind": "detach",
        "guard_set": _GUARD_SET,
        "name_ids": name_ids,
        "reason": _REASON,
        "gc": graph,
    }
    return apply_signed_manifest(path, **{**arguments, **overrides})


def test_program_is_registered_in_the_signed_registry() -> None:
    assert operator._RECOVERY_UNBIND_GUARDS == _GUARD_SET
    assert operator._GUARD_KINDS[operator._EXHAUSTED_TARGET] == "semantic_authority"


def test_exhausted_row_loses_its_binding_and_records_a_change(tmp_path: Path) -> None:
    graph = FakeGraph()
    graph.add_name("recovered_name", "exhausted", unit="T")
    graph.add_source("dd:path/to/value")
    graph.bind("dd:path/to/value", "recovered_name")
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "recovered_name", "source_id": "dd:path/to/value"}])

    preview = _apply(graph, path, ["recovered_name"])

    assert preview["outcome"] == "would_apply"
    assert preview["counts"] == {"authority_rows": 1, "admitted": 1, "refused": 0}
    assert preview["manifest"]["admitted_row_ids"] == ["recovered_name"]
    assert graph.snapshot()["bindings"]

    applied = _apply(
        graph,
        path,
        ["recovered_name"],
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )

    assert applied["outcome"] == "applied"
    assert applied["changed"] == 1
    assert graph.snapshot()["bindings"] == []
    change = graph.changes[applied["change_ids"][0]]
    assert change["to_name"] == "recovered_name"
    assert change["from_name"] == "dd:path/to/value"
    assert change["operation"] == operator._RECOVERY_UNBIND_MUTATION
    assert change["manifest_sha256"] == preview["manifest_sha256"]


def test_unbind_leaves_the_name_stage_edges_and_source_intact(tmp_path: Path) -> None:
    graph = FakeGraph()
    graph.add_name("recovered_name", "exhausted", unit="T", physics_domain="magnetics")
    graph.add_name("sibling_name", "accepted")
    graph.add_source("dd:path/to/value", status="composed")
    graph.bind("dd:path/to/value", "recovered_name")
    graph.bind("dd:path/to/value", "sibling_name")
    before_names = deepcopy(graph.names)
    before_sources = deepcopy(graph.sources)
    before_siblings = [
        binding for binding in graph.bindings if binding["name_id"] == "sibling_name"
    ]
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "recovered_name", "source_id": "dd:path/to/value"}])

    preview = _apply(graph, path, ["recovered_name"])
    _apply(
        graph,
        path,
        ["recovered_name"],
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )

    assert graph.names == before_names
    assert graph.sources == before_sources
    assert [
        binding for binding in graph.bindings if binding["name_id"] == "sibling_name"
    ] == before_siblings


def test_live_name_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    graph = FakeGraph()
    graph.add_name("live_name", "accepted")
    graph.add_source("dd:path/to/value")
    graph.bind("dd:path/to/value", "live_name")
    before = graph.snapshot()
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "live_name", "source_id": "dd:path/to/value"}])

    preview = _apply(graph, path, ["live_name"])

    assert preview["outcome"] == "refused"
    assert preview["refusals"] == [
        {"row_id": "live_name", "reason": "target name is not exhausted"}
    ]
    assert graph.snapshot() == before

    refused = _apply(
        graph,
        path,
        ["live_name"],
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )

    assert refused["outcome"] == "refused"
    assert refused["changed"] == 0
    assert graph.snapshot() == before


def test_missing_binding_is_refused(tmp_path: Path) -> None:
    graph = FakeGraph()
    graph.add_name("recovered_name", "exhausted")
    graph.add_source("dd:path/to/value")
    before = graph.snapshot()
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "recovered_name", "source_id": "dd:path/to/value"}])

    preview = _apply(graph, path, ["recovered_name"])

    assert preview["outcome"] == "refused"
    assert preview["refusals"] == [
        {"row_id": "recovered_name", "reason": "recovery binding does not exist"}
    ]
    assert graph.snapshot() == before


def test_stale_source_is_refused(tmp_path: Path) -> None:
    graph = FakeGraph()
    graph.add_name("recovered_name", "exhausted")
    graph.add_source("dd:path/to/value", status="stale")
    graph.bind("dd:path/to/value", "recovered_name")
    before = graph.snapshot()
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "recovered_name", "source_id": "dd:path/to/value"}])

    preview = _apply(graph, path, ["recovered_name"])

    assert preview["outcome"] == "refused"
    assert preview["refusals"] == [
        {"row_id": "recovered_name", "reason": "source is stale"}
    ]
    assert graph.snapshot() == before


def test_apply_refuses_a_manifest_that_is_not_authorized(tmp_path: Path) -> None:
    graph = FakeGraph()
    graph.add_name("recovered_name", "exhausted")
    graph.add_source("dd:path/to/value")
    graph.bind("dd:path/to/value", "recovered_name")
    before = graph.snapshot()
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "recovered_name", "source_id": "dd:path/to/value"}])
    _apply(graph, path, ["recovered_name"])

    with pytest.raises(operator.RecoveryUnbindConflict):
        _apply(
            graph,
            path,
            ["recovered_name"],
            apply=True,
            manifest_sha256="0" * 64,
        )
    assert graph.snapshot() == before


def test_replay_after_apply_is_write_free(tmp_path: Path) -> None:
    graph = FakeGraph()
    graph.add_name("recovered_name", "exhausted")
    graph.add_source("dd:path/to/value")
    graph.bind("dd:path/to/value", "recovered_name")
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "recovered_name", "source_id": "dd:path/to/value"}])
    preview = _apply(graph, path, ["recovered_name"])
    _apply(
        graph,
        path,
        ["recovered_name"],
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )
    after = graph.snapshot()

    replay = _apply(
        graph,
        path,
        ["recovered_name"],
        apply=True,
        manifest_sha256=preview["manifest_sha256"],
    )

    assert replay["outcome"] == "already_applied"
    assert replay["changed"] == 0
    assert graph.snapshot() == after


def test_unbind_requires_its_exact_signed_guard_set(tmp_path: Path) -> None:
    graph = FakeGraph()
    path = tmp_path / "authority.json"
    _write_authority(path, [{"name_id": "recovered_name", "source_id": "dd:path/to/value"}])

    with pytest.raises(SignedManifestAuthorityError):
        _apply(graph, path, ["recovered_name"], guard_set=("exhausted-target-live-source",))


def test_source_invariant_still_refuses_a_sourceless_live_name() -> None:
    graph = FakeGraph()
    graph.add_name("live_name", "accepted")
    graph.add_name("recovered_name", "exhausted")

    with pytest.raises(MissingStandardNameSource, match="live_name"):
        require_producing_sources(graph, ["live_name"])

    require_producing_sources(
        graph, ["recovered_name"], exhausted_exempt_ids=["recovered_name"]
    )


def test_exemption_is_confined_to_its_own_exhausted_rows() -> None:
    graph = FakeGraph()
    graph.add_name("live_name", "accepted")

    # A live name named in the exemption set is still refused, so the exemption
    # can never excuse a name the graph has since moved on from.
    with pytest.raises(MissingStandardNameSource, match="live_name"):
        require_producing_sources(
            graph, ["live_name"], exhausted_exempt_ids=["live_name"]
        )
