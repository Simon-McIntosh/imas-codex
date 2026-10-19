"""A signed DD source exchange changes producer ownership atomically."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from imas_codex.standard_names import signed_manifest as operator
from imas_codex.standard_names.repair_authority import build_repair_authority
from tests.standard_names.test_signed_recovery_unbind import (
    FakeGraph as _BaseGraph,
    _Transaction as _BaseTransaction,
)


class _Transaction(_BaseTransaction):
    def __init__(self, state: Graph) -> None:
        super().__init__(state)
        self.before = state.snapshot()

    def rollback(self) -> None:
        self.state.names = deepcopy(self.before["names"])
        self.state.sources = deepcopy(self.before["sources"])
        self.state.bindings = deepcopy(self.before["bindings"])
        self.state.changes = deepcopy(self.before["changes"])
        super().rollback()

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append(query)
        if "SOURCE_EXCHANGE_REPLAY" in query:
            return [
                {"event_id": event_id, "recorded": event_id in self.state.changes}
                for event_id in params["event_ids"]
            ]
        if "SOURCE_EXCHANGE_CURRENT_DD" in query:
            return [{"dd_version": "4.0.0"}]
        if "SOURCE_EXCHANGE_CLOSURE" in query:
            result = []
            for row in params["rows"]:
                name_id = row["name_id"]
                old_id = row["given_source_id"]
                new_id = row["taken_source_id"]
                receiver_id = row["receiving_name_id"]
                name = self.state.names.get(name_id)
                old = self.state.sources.get(old_id)
                new = self.state.sources.get(new_id)
                receiver = self.state.names.get(receiver_id)
                old_dd = self.state.dd_paths.get(old_id[3:])
                new_dd = self.state.dd_paths.get(new_id[3:])
                result.append(
                    {
                        "name_id": name_id,
                        "name_element_id": name and name["element_id"],
                        "name_stage": name and name["properties"].get("name_stage"),
                        "given_source_id": old_id,
                        "given_element_id": old and old["element_id"],
                        "given_properties": old and deepcopy(old["properties"]),
                        "given_holders": [
                            {
                                "name_id": binding["name_id"],
                                "binding_id": binding["element_id"],
                            }
                            for binding in self.state.bindings
                            if binding["source_id"] == old_id
                        ],
                        "taken_source_id": new_id,
                        "taken_element_id": new and new["element_id"],
                        "taken_properties": new and deepcopy(new["properties"]),
                        "taken_holders": [
                            {
                                "name_id": binding["name_id"],
                                "stage": self.state.names[binding["name_id"]][
                                    "properties"
                                ].get("name_stage"),
                            }
                            for binding in self.state.bindings
                            if binding["source_id"] == new_id
                        ],
                        "taken_dd_element_id": new_dd and f"dd:{new_id[3:]}",
                        "taken_dd_lifecycle_status": new_dd,
                        "given_dd_element_id": old_dd and f"dd:{old_id[3:]}",
                        "given_dd_lifecycle_status": old_dd,
                        "receiving_name_id": receiver_id,
                        "receiving_element_id": receiver and receiver["element_id"],
                        "receiving_stage": receiver
                        and receiver["properties"].get("name_stage"),
                    }
                )
            return result
        if "SOURCE_EXCHANGE_MUTATION" in query:
            old = self.state.sources[params["given_source_id"]]
            name_id = params["name_id"]
            binding = self._binding(params["given_source_id"], name_id)
            if not binding or binding["element_id"] != params["binding_element_id"]:
                return [{"changed": 0}]
            new_id = params["taken_source_id"]
            self.state.sources.setdefault(
                new_id, {"element_id": f"src:{new_id}", "properties": {"id": new_id}}
            )["properties"].update({"status": "attached", "produced_sn_id": name_id})
            self.state.bindings.remove(binding)
            self.state.bind(new_id, name_id)
            receiver_id = params["receiving_name_id"]
            old["properties"]["produced_sn_id"] = receiver_id
            old["properties"]["status"] = "attached" if receiver_id else "extracted"
            if receiver_id:
                self.state.bind(params["given_source_id"], receiver_id)
            return [{"changed": 1}]
        if "SOURCE_EXCHANGE_RECEIPT" in query:
            for row in params["rows"]:
                self.state.changes[row["event_id"]] = {
                    "id": row["event_id"],
                    "manifest_sha256": params["manifest_sha256"],
                }
            return [{"event_id": row["event_id"]} for row in params["rows"]]
        if "SOURCE_EXCHANGE_NAME_POSTCONDITION" in query:
            return [
                {
                    "name_id": name_id,
                    "producers": (
                        0
                        if self.state.report_missing_name == name_id
                        else sum(
                            binding["name_id"] == name_id
                            for binding in self.state.bindings
                        )
                    ),
                }
                for name_id in params["name_ids"]
            ]
        if "SOURCE_EXCHANGE_SOURCE_POSTCONDITION" in query:
            return [
                {
                    "source_id": source_id,
                    "producers": sum(
                        binding["source_id"] == source_id
                        for binding in self.state.bindings
                    ),
                    "names": [
                        binding["name_id"]
                        for binding in self.state.bindings
                        if binding["source_id"] == source_id
                    ],
                }
                for source_id in params["source_ids"]
            ]
        return super().run(query, **params)


class Graph(_BaseGraph):
    def __init__(self) -> None:
        super().__init__()
        self.dd_paths: dict[str, str] = {}
        self.report_missing_name: str | None = None

    def add_dd(self, path: str, lifecycle: str = "active") -> None:
        self.dd_paths[path] = lifecycle

    def add_source(self, source_id: str, status: str = "composed") -> None:
        super().add_source(source_id, status)
        self.sources[source_id]["properties"].update(
            {
                "source_type": "dd",
                "source_id": source_id[3:],
                "dd_path": source_id[3:],
            }
        )

    def session(self) -> Any:
        from tests.standard_names.test_signed_recovery_unbind import _Session

        transaction = _Transaction(self)
        self.transactions.append(transaction)
        return _Session(transaction)


def _authority(path: Path, *, receiver: str | None = None) -> tuple[str, str]:
    row: dict[str, Any] = {
        "id": "local_temperature",
        "name_id": "local_temperature",
        "given_source_id": "dd:summary/average/value",
        "taken_source_id": "dd:summary/local/value",
    }
    if receiver:
        row["receiving_name_id"] = receiver
    built = build_repair_authority(
        {
            "operation_id": operator._SOURCE_EXCHANGE_OPERATION,
            "authority_mode": "external_reviewed",
            "rows": [row],
            "receipt_policy": {
                "id": "one-per-logical-change",
                "operation": "exchange_dd_producing_source",
                "cardinality": "per_target",
                "link_participant_kind": "standard_name",
                "replay_projection": ["manifest_sha256", "row_id"],
            },
            "orphan_policy": "refuse",
        }
    )
    path.write_bytes(built.content)
    assert json.loads(path.read_bytes())["signature"]["sha256"] == built.payload_sha256
    return built.file_sha256, built.payload_sha256


def _graph(*, receiver: str | None = None) -> Graph:
    graph = Graph()
    graph.add_name("local_temperature", "accepted")
    if receiver:
        graph.add_name(receiver, "accepted")
    graph.add_dd("summary/average/value")
    graph.add_dd("summary/local/value")
    graph.add_source("dd:summary/average/value")
    graph.bind("dd:summary/average/value", "local_temperature")
    return graph


def _run(
    graph: Graph, path: Path, digests: tuple[str, str], **kw: Any
) -> dict[str, Any]:
    arguments = {
        "authority_adapter": operator._SOURCE_EXCHANGE_ADAPTER,
        "authority_file_sha256": digests[0],
        "authority_payload_sha256": digests[1],
        "guard_set": operator._SOURCE_EXCHANGE_GUARDS,
        "reason": "correct the DD ownership",
        "gc": graph,
    }
    return operator.apply_signed_manifest(path, **{**arguments, **kw})


@pytest.mark.parametrize("receiver", [None, "average_temperature"])
def test_signed_exchange_is_atomic_and_records_receipt(
    tmp_path: Path, receiver: str | None
) -> None:
    graph = _graph(receiver=receiver)
    graph.add_source("dd:summary/local/value", status="extracted")
    graph.sources["dd:summary/local/value"]["properties"]["dd_version"] = "4.0.0"
    path = tmp_path / "authority.json"
    digests = _authority(path, receiver=receiver)
    before = graph.snapshot()
    preview = _run(graph, path, digests)
    assert preview["outcome"] == "would_apply"
    assert graph.snapshot() == before
    applied = _run(
        graph, path, digests, apply=True, manifest_sha256=preview["manifest_sha256"]
    )
    assert applied["outcome"] == "applied"
    assert any(
        binding["source_id"] == "dd:summary/local/value"
        and binding["name_id"] == "local_temperature"
        for binding in graph.bindings
    )
    assert any(
        binding["source_id"] == "dd:summary/average/value"
        and binding["name_id"] == receiver
        for binding in graph.bindings
    ) == bool(receiver)
    assert len(graph.changes) == 1
    assert (
        graph.sources["dd:summary/local/value"]["properties"]["dd_version"] == "4.0.0"
    )
    assert graph.sources["dd:summary/average/value"]["properties"]["status"] == (
        "attached" if receiver else "extracted"
    )
    assert graph.transactions[-1].committed


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("exhausted_sender", "giving name is absent or not live"),
        ("superseded_sender", "giving name is absent or not live"),
        ("wrong_owner", "given source does not produce exactly the giving name"),
        ("missing_given", "given source does not produce exactly the giving name"),
        ("missing_taken_dd", "taken DD path is absent or removed from the current DD"),
        ("removed_taken_dd", "taken DD path is absent or removed from the current DD"),
        ("live_taken_owner", "taken source already has a live owner"),
        ("exhausted_receiver", "receiving name is absent or not live"),
        ("removed_given_dd", "given DD path is removed from the current DD"),
    ],
)
def test_invalid_exchange_refuses_before_writing(
    tmp_path: Path, defect: str, reason: str
) -> None:
    receiver = (
        "average_temperature"
        if defect in {"exhausted_receiver", "removed_given_dd"}
        else None
    )
    graph = _graph(receiver=receiver)
    if defect == "exhausted_sender":
        graph.names["local_temperature"]["properties"]["name_stage"] = "exhausted"
    elif defect == "superseded_sender":
        graph.names["local_temperature"]["properties"]["name_stage"] = "superseded"
    elif defect == "wrong_owner":
        graph.add_name("other", "accepted")
        graph.bindings[0]["name_id"] = "other"
    elif defect == "missing_given":
        graph.bindings.clear()
    elif defect == "missing_taken_dd":
        del graph.dd_paths["summary/local/value"]
    elif defect == "removed_taken_dd":
        graph.dd_paths["summary/local/value"] = "removed"
    elif defect == "live_taken_owner":
        graph.add_name("other", "accepted")
        graph.add_source("dd:summary/local/value")
        graph.bind("dd:summary/local/value", "other")
    elif defect == "exhausted_receiver":
        graph.names["average_temperature"]["properties"]["name_stage"] = "exhausted"
    elif defect == "removed_given_dd":
        graph.dd_paths["summary/average/value"] = "removed"
    path = tmp_path / "authority.json"
    digests = _authority(path, receiver=receiver)
    before = graph.snapshot()
    receipt = _run(graph, path, digests)
    assert receipt["outcome"] == "refused"
    assert receipt["refusals"] == [{"row_id": "local_temperature", "reason": reason}]
    assert graph.snapshot() == before
    applied = _run(
        graph, path, digests, apply=True, manifest_sha256=receipt["manifest_sha256"]
    )
    assert applied["outcome"] == "refused"
    assert graph.snapshot() == before


def test_failed_name_postcondition_rolls_back_every_write(tmp_path: Path) -> None:
    graph = _graph(receiver="average_temperature")
    path = tmp_path / "authority.json"
    digests = _authority(path, receiver="average_temperature")
    preview = _run(graph, path, digests)
    before = graph.snapshot()
    graph.report_missing_name = "average_temperature"
    with pytest.raises(operator.SourceExchangeConflict, match="touched name lost"):
        _run(
            graph, path, digests, apply=True, manifest_sha256=preview["manifest_sha256"]
        )
    assert graph.transactions[-1].rolled_back
    assert graph.snapshot() == before


def test_applied_row_replay_is_refused_without_writes(tmp_path: Path) -> None:
    graph = _graph()
    path = tmp_path / "authority.json"
    digests = _authority(path)
    preview = _run(graph, path, digests)
    _run(graph, path, digests, apply=True, manifest_sha256=preview["manifest_sha256"])
    after = graph.snapshot()
    replay = _run(
        graph, path, digests, apply=True, manifest_sha256=preview["manifest_sha256"]
    )
    assert replay["outcome"] == "refused"
    assert replay["refusals"][0]["reason"] == "signed source exchange already applied"
    assert graph.snapshot() == after


def test_extra_taken_source_owner_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = _graph()
    graph.add_name("other", "accepted")
    path = tmp_path / "authority.json"
    digests = _authority(path)
    preview = _run(graph, path, digests)
    before = graph.snapshot()
    original = operator._verify_source_exchange_postcondition

    def duplicate_owner(query: Any, rows: Any) -> None:
        graph.bind("dd:summary/local/value", "other")
        original(query, rows)

    monkeypatch.setattr(
        operator, "_verify_source_exchange_postcondition", duplicate_owner
    )
    with pytest.raises(operator.SourceExchangeConflict, match="touched source"):
        _run(
            graph, path, digests, apply=True, manifest_sha256=preview["manifest_sha256"]
        )
    assert graph.transactions[-1].rolled_back
    assert graph.snapshot() == before


def test_authority_digest_and_guard_set_are_required(tmp_path: Path) -> None:
    graph = _graph()
    path = tmp_path / "authority.json"
    digests = _authority(path)
    with pytest.raises(operator.SignedManifestAuthorityError, match="guard set"):
        _run(graph, path, digests, guard_set=())
    with pytest.raises(operator.SignedManifestAuthorityError, match="file SHA-256"):
        _run(graph, path, ("0" * 64, digests[1]))
