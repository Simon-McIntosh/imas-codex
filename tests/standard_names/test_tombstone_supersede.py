"""Stateful transaction coverage for folding one identity into another."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest

from imas_codex.standard_names import edit


def _node(
    name: str,
    *,
    stage: str,
    validation: str = "valid",
    unit: str = "m^-3",
    paths: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": name,
        "name_stage": stage,
        "validation_status": validation,
        "unit": unit,
        "source_paths": list(paths or []),
        "claim_token": None,
        "claimed_at": None,
        **extra,
    }


@dataclass
class _State:
    nodes: dict[str, dict[str, Any]]
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    backings: dict[str, dict[str, Any]] = field(default_factory=dict)
    refined_from: list[tuple[str, str]] = field(default_factory=list)
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    change_links: list[tuple[str, str]] = field(default_factory=list)
    reviews: dict[str, dict[str, Any]] = field(default_factory=dict)
    review_links: list[tuple[str, str]] = field(default_factory=list)
    revisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    revision_links: list[tuple[str, str]] = field(default_factory=list)
    other_relationships: list[dict[str, Any]] = field(default_factory=list)
    unrelated: dict[str, Any] = field(default_factory=lambda: {"sentinel": [1, 2, 3]})


class _Transaction:
    """Copy-on-write graph transaction that mirrors the production queries."""

    def __init__(self, graph: _Graph) -> None:
        self.graph = graph
        self.state = copy.deepcopy(graph.state)
        self.committed = False
        self.rolled_back = False
        self.write_markers: list[str] = []
        self.snapshot_count = 0

    @staticmethod
    def _element_id(kind: str, identifier: str) -> str:
        return f"{kind}:{identifier}"

    @staticmethod
    def _relationship(
        kind: str,
        key: str,
        start_kind: str,
        start_id: str,
        end_kind: str,
        end_id: str,
        *,
        properties: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "element_id": f"rel:{kind}:{key}",
            "type": kind,
            "start_element_id": f"{start_kind}:{start_id}",
            "end_element_id": f"{end_kind}:{end_id}",
            "start_id": start_id,
            "end_id": end_id,
            "start_labels": [_label(start_kind)],
            "end_labels": [_label(end_kind)],
            "properties": copy.deepcopy(properties or {}),
        }

    def _all_relationships(self) -> list[dict[str, Any]]:
        relationships: list[dict[str, Any]] = []
        for source_id, source in self.state.sources.items():
            for index, bound in enumerate(source["bindings"]):
                relationships.append(
                    self._relationship(
                        "PRODUCED_NAME",
                        f"{source_id}:{index}:{bound}",
                        "source",
                        source_id,
                        "name",
                        bound,
                    )
                )
            for index, backing_id in enumerate(source.get("backings", [])):
                kind = (
                    "FROM_DD_PATH"
                    if "IMASNode" in self.state.backings[backing_id]["labels"]
                    else "FROM_SIGNAL"
                )
                relationships.append(
                    self._relationship(
                        kind,
                        f"{source_id}:{index}:{backing_id}",
                        "source",
                        source_id,
                        "backing",
                        backing_id,
                    )
                )
        for backing_id, backing in self.state.backings.items():
            for index, projected in enumerate(backing["projections"]):
                relationships.append(
                    self._relationship(
                        "HAS_STANDARD_NAME",
                        f"{backing_id}:{index}:{projected}",
                        "backing",
                        backing_id,
                        "name",
                        projected,
                    )
                )
            for index, unit in enumerate(backing.get("units", [])):
                relationships.append(
                    self._relationship(
                        "HAS_UNIT",
                        f"{backing_id}:{index}:{unit}",
                        "backing",
                        backing_id,
                        "unit",
                        unit,
                    )
                )
        for index, (successor, predecessor) in enumerate(self.state.refined_from):
            relationships.append(
                self._relationship(
                    "REFINED_FROM",
                    f"{index}:{successor}:{predecessor}",
                    "name",
                    successor,
                    "name",
                    predecessor,
                )
            )
        for index, (owner, change_id) in enumerate(self.state.change_links):
            relationships.append(
                self._relationship(
                    "HAS_INTERNAL_CHANGE",
                    f"{index}:{owner}:{change_id}",
                    "name",
                    owner,
                    "change",
                    change_id,
                )
            )
        for index, (owner, review_id) in enumerate(self.state.review_links):
            relationships.append(
                self._relationship(
                    "HAS_REVIEW",
                    f"{index}:{owner}:{review_id}",
                    "name",
                    owner,
                    "review",
                    review_id,
                )
            )
        for index, (owner, revision_id) in enumerate(self.state.revision_links):
            relationships.append(
                self._relationship(
                    "DOCS_REVISION_OF",
                    f"{index}:{owner}:{revision_id}",
                    "name",
                    owner,
                    "revision",
                    revision_id,
                )
            )
        relationships.extend(copy.deepcopy(self.state.other_relationships))
        return relationships

    def _incident_relationships(self, old: str, target: str) -> list[dict[str, Any]]:
        name_ids = {
            self._element_id("name", old),
            self._element_id("name", target),
        }
        relationships = []
        for relationship in self._all_relationships():
            if (
                relationship["start_element_id"] not in name_ids
                and relationship["end_element_id"] not in name_ids
            ):
                continue
            item = copy.deepcopy(relationship)
            item["other_element_id"] = (
                item["end_element_id"]
                if item["start_element_id"] in name_ids
                else item["start_element_id"]
            )
            relationships.append(item)
        return relationships

    def _descends(self, successor: str, predecessor: str) -> bool:
        seen: set[str] = set()
        frontier = [successor]
        while frontier:
            current = frontier.pop()
            for child, parent in self.state.refined_from:
                if child != current or parent in seen:
                    continue
                if parent == predecessor:
                    return True
                seen.add(parent)
                frontier.append(parent)
        return False

    def _candidate_source_ids(self, old: str, target: str) -> set[str]:
        ids = set()
        for source_id, source in self.state.sources.items():
            projected = {
                name
                for backing_id in source.get("backings", [])
                for name in self.state.backings[backing_id]["projections"]
            }
            if (
                source["properties"].get("produced_sn_id") in {old, target}
                or set(source["bindings"]) & {old, target}
                or projected & {old, target}
            ):
                ids.add(source_id)
        return ids

    def _source_row(self, source_id: str) -> dict[str, Any]:
        source = self.state.sources[source_id]
        scalar_id = source["properties"].get("produced_sn_id")
        scalar = self.state.nodes.get(scalar_id)
        return {
            "id": source_id,
            "element_id": self._element_id("source", source_id),
            "labels": ["StandardNameSource"],
            "properties": copy.deepcopy(source["properties"]),
            "scalar_target": (
                {
                    "element_id": self._element_id("name", scalar_id),
                    "labels": ["StandardName"],
                    "properties": copy.deepcopy(scalar),
                    "target_id": scalar_id,
                    "target_stage": scalar["name_stage"],
                }
                if scalar is not None
                else None
            ),
            "bindings": [
                {
                    "element_id": f"rel:PRODUCED_NAME:{source_id}:{index}:{target}",
                    "properties": {},
                    "target_element_id": self._element_id("name", target),
                    "target_labels": ["StandardName"],
                    "target_properties": copy.deepcopy(self.state.nodes[target]),
                    "target_id": target,
                    "target_stage": self.state.nodes[target]["name_stage"],
                }
                for index, target in enumerate(source["bindings"])
            ],
            "backing_refs": [
                {
                    "element_id": f"rel:owner:{source_id}:{index}:{backing_id}",
                    "properties": {},
                    "type": (
                        "FROM_DD_PATH"
                        if "IMASNode" in self.state.backings[backing_id]["labels"]
                        else "FROM_SIGNAL"
                    ),
                    "backing_element_id": self._element_id("backing", backing_id),
                    "backing_id": backing_id,
                }
                for index, backing_id in enumerate(source.get("backings", []))
            ],
        }

    def _backing_row(self, backing_id: str) -> dict[str, Any]:
        backing = self.state.backings[backing_id]
        owners = [
            (source_id, index)
            for source_id, source in self.state.sources.items()
            for index, owned in enumerate(source.get("backings", []))
            if owned == backing_id
        ]
        return {
            "id": backing_id,
            "element_id": self._element_id("backing", backing_id),
            "labels": list(backing["labels"]),
            "properties": copy.deepcopy(backing["properties"]),
            "owners": [
                {
                    "source_id": source_id,
                    "source_element_id": self._element_id("source", source_id),
                    "relationship_element_id": (
                        f"rel:owner:{source_id}:{index}:{backing_id}"
                    ),
                    "relationship_properties": {},
                    "relationship_type": (
                        "FROM_DD_PATH"
                        if "IMASNode" in backing["labels"]
                        else "FROM_SIGNAL"
                    ),
                }
                for source_id, index in owners
            ],
            "projections": [
                {
                    "element_id": (
                        f"rel:HAS_STANDARD_NAME:{backing_id}:{index}:{target}"
                    ),
                    "properties": {},
                    "target_element_id": self._element_id("name", target),
                    "target_labels": ["StandardName"],
                    "target_properties": copy.deepcopy(self.state.nodes[target]),
                    "target_id": target,
                    "target_stage": self.state.nodes[target]["name_stage"],
                }
                for index, target in enumerate(backing["projections"])
            ],
            "units": [
                {
                    "element_id": f"rel:HAS_UNIT:{backing_id}:{index}:{unit}",
                    "properties": {},
                    "unit_element_id": self._element_id("unit", unit),
                    "unit_labels": ["Unit"],
                    "unit_id": unit,
                    "unit_properties": {"id": unit},
                }
                for index, unit in enumerate(backing.get("units", []))
            ],
        }

    def _owned_record(
        self,
        kind: str,
        identifier: str,
        properties: dict[str, Any],
        links: list[tuple[str, str]],
        relationship: str,
    ) -> dict[str, Any]:
        return {
            "element_id": self._element_id(kind, identifier),
            "labels": [_label(kind)],
            "properties": copy.deepcopy(properties),
            "owners": [
                {
                    "owner_id": owner,
                    "element_id": f"rel:{relationship}:{index}:{owner}:{identifier}",
                    "properties": {},
                }
                for index, (owner, owned) in enumerate(links)
                if owned == identifier
            ],
        }

    def _snapshot(self, old: str, target: str) -> list[dict[str, Any]]:
        self.snapshot_count += 1
        if old not in self.state.nodes or target not in self.state.nodes:
            return []
        candidate_sources = self._candidate_source_ids(old, target)
        candidate_backings = {
            backing_id
            for backing_id, backing in self.state.backings.items()
            if set(backing["projections"]) & {old, target}
            or any(
                source_id in candidate_sources
                and backing_id in self.state.sources[source_id].get("backings", [])
                for source_id in self.state.sources
            )
        }
        reviews = [
            self._owned_record(
                "review",
                review_id,
                properties,
                self.state.review_links,
                "HAS_REVIEW",
            )
            for review_id, properties in self.state.reviews.items()
            if properties.get("standard_name_id") in {old, target}
            or any(
                owner in {old, target} and owned == review_id
                for owner, owned in self.state.review_links
            )
        ]
        revisions = [
            self._owned_record(
                "revision",
                revision_id,
                properties,
                self.state.revision_links,
                "DOCS_REVISION_OF",
            )
            for revision_id, properties in self.state.revisions.items()
            if any(
                owner in {old, target} and owned == revision_id
                for owner, owned in self.state.revision_links
            )
        ]
        changes = [
            self._owned_record(
                "change",
                change_id,
                properties,
                self.state.change_links,
                "HAS_INTERNAL_CHANGE",
            )
            for change_id, properties in self.state.changes.items()
            if any(
                owner in {old, target} and owned == change_id
                for owner, owned in self.state.change_links
            )
        ]
        return [
            {
                "old_element_id": self._element_id("name", old),
                "target_element_id": self._element_id("name", target),
                "old_labels": ["StandardName"],
                "target_labels": ["StandardName"],
                "old_properties": copy.deepcopy(self.state.nodes[old]),
                "target_properties": copy.deepcopy(self.state.nodes[target]),
                "cycle": self._descends(old, target),
                "sources": [self._source_row(value) for value in candidate_sources],
                "backings": [self._backing_row(value) for value in candidate_backings],
                "relationships": self._incident_relationships(old, target),
                "reviews": reviews,
                "revisions": revisions,
                "changes": changes,
                "old_units": self._name_units(old),
                "target_units": self._name_units(target),
            }
        ]

    def _name_units(self, name: str) -> list[dict[str, Any]]:
        return [
            {
                "element_id": f"rel:HAS_UNIT:{name}:{index}:{unit}",
                "properties": {},
                "unit_element_id": self._element_id("unit", unit),
                "unit_labels": ["Unit"],
                "unit_id": unit,
                "unit_properties": {"id": unit},
            }
            for index, unit in enumerate(
                self.state.nodes[name].get("relationship_units", [])
            )
        ]

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "ATOMIC_FOLD_SNAPSHOT" in cypher or "ATOMIC_FOLD_POSTFLIGHT" in cypher:
            return self._snapshot(params["old_id"], params["into_id"])
        if "ATOMIC_FOLD_EVENT" in cypher:
            self.write_markers.append("event")
            if self.graph.fail_at == "event":
                raise RuntimeError("injected event failure")
            old = self.state.nodes[params["old_id"]]
            target = self.state.nodes[params["into_id"]]
            if old != params["old_properties"] or target != params["target_properties"]:
                return []
            change = {
                "id": params["change_id"],
                "from_name": params["old_id"],
                "to_name": params["into_id"],
                "operation": "fold_identity",
                "reason": params["receipt"],
                "origin": "catalog_edit",
                "run_id": params["run_id"],
                "changed_at": params["changed_at"],
                "internal": True,
            }
            self.state.changes[change["id"]] = change
            self.state.change_links.extend(
                [
                    (params["old_id"], change["id"]),
                    (params["into_id"], change["id"]),
                ]
            )
            return [{"change_id": change["id"]}]
        if "ATOMIC_FOLD_LOCK" in cypher:
            self.write_markers.append("lock")
            if self.graph.fail_at == "race":
                self.state.nodes[self.graph.target]["documentation"] = "concurrent"
            elif self.graph.fail_at == "competitor_race":
                self.state.nodes["retired_density"]["name_stage"] = "accepted"
            elif self.graph.fail_at == "unit_race":
                next(iter(self.state.backings.values()))["units"] = ["K"]
            return [{"locked": len(params["element_ids"])}]
        if "RETURN source_id," in cypher and "already_bound" in cypher:
            target = params["sn_id"]
            existing = sorted(
                {
                    backing_id
                    for source in self.state.sources.values()
                    if target in source["bindings"]
                    for backing_id in source.get("backings", [])
                    if "IMASNode" in self.state.backings[backing_id]["labels"]
                }
            )
            rows = []
            for source_id in params["source_ids"]:
                source = self.state.sources[source_id]
                backing_ids = source.get("backings", [])
                backing = self.state.backings[backing_ids[0]] if backing_ids else None
                is_dd = backing is not None and "IMASNode" in backing["labels"]
                rows.append(
                    {
                        "source_id": source_id,
                        "source_type": source["properties"]["source_type"],
                        "dd_path": backing["properties"]["id"] if is_dd else None,
                        "dd_unit": (
                            backing["units"][0]
                            if is_dd and backing.get("units")
                            else backing["properties"].get("unit")
                            if is_dd
                            else None
                        ),
                        "sn_unit": self.state.nodes[target]["unit"],
                        "already_bound": target in source["bindings"],
                        "existing_dd_paths": existing,
                        "name_stage": self.state.nodes[target]["name_stage"],
                    }
                )
            return rows
        if "ATOMIC_FOLD_MOVE_SOURCES" in cypher:
            self.write_markers.append("sources")
            moved_sources = 0
            for index, expected in enumerate(params["sources"]):
                source = self.state.sources[expected["id"]]
                if (
                    self._element_id("source", expected["id"]) != expected["element_id"]
                    or source["properties"] != expected["properties"]
                ):
                    continue
                remove = set(expected["remove_binding_element_ids"])
                source["bindings"] = [
                    name
                    for binding_index, name in enumerate(source["bindings"])
                    if f"rel:PRODUCED_NAME:{expected['id']}:{binding_index}:{name}"
                    not in remove
                ]
                source["bindings"].append(params["into_id"])
                source["properties"]["produced_sn_id"] = params["into_id"]
                moved_sources += 1
                if self.graph.fail_at == "partial_source" and index == 0:
                    raise RuntimeError("injected partial source migration")
            moved_backings = 0
            for expected in params["backings"]:
                backing_id = expected["properties"]["id"]
                backing = self.state.backings[backing_id]
                if backing["properties"] != expected["properties"]:
                    continue
                remove = set(expected["remove_projection_element_ids"])
                backing["projections"] = [
                    name
                    for projection_index, name in enumerate(backing["projections"])
                    if (f"rel:HAS_STANDARD_NAME:{backing_id}:{projection_index}:{name}")
                    not in remove
                ]
                backing["projections"].append(params["into_id"])
                if expected["has_standard_name_id"]:
                    backing["properties"]["standard_name_id"] = params["into_id"]
                moved_backings += 1
            return [
                {
                    "sources_moved": moved_sources,
                    "projections_moved": moved_backings,
                }
            ]
        if "ATOMIC_FOLD_MUTATE_NAMES" in cypher:
            self.write_markers.append("names")
            old = self.state.nodes[params["old_id"]]
            target = self.state.nodes[params["into_id"]]
            if old != params["old_properties"] or target != params["target_properties"]:
                return []
            old["superseded_from_stage"] = params["predecessor_stage"]
            old["name_stage"] = "superseded"
            old["claim_token"] = None
            old["claimed_at"] = None
            old["source_paths"] = []
            if old.get("edit_status") == "open":
                old["edit_status"] = "applied"
            target["source_paths"] = list(params["target_paths"])
            lineage = (params["into_id"], params["old_id"])
            if lineage not in self.state.refined_from:
                self.state.refined_from.append(lineage)
            return [
                {
                    "old_stage": "superseded",
                    "predecessor_stage": params["predecessor_stage"],
                }
            ]
        raise AssertionError(f"unexpected query: {cypher}")

    def commit(self) -> None:
        self.graph.state = self.state
        self.graph.commits += 1
        self.committed = True

    def rollback(self) -> None:
        self.graph.rollbacks += 1
        self.rolled_back = True


def _label(kind: str) -> str:
    return {
        "name": "StandardName",
        "source": "StandardNameSource",
        "backing": "IMASNode",
        "unit": "Unit",
        "change": "StandardNameChange",
        "review": "StandardNameReview",
        "revision": "DocsRevision",
    }[kind]


class _Session:
    def __init__(self, graph: _Graph) -> None:
        self.graph = graph

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def begin_transaction(self) -> _Transaction:
        transaction = _Transaction(self.graph)
        self.graph.transactions.append(transaction)
        return transaction


class _Graph:
    def __init__(self, state: _State, *, fail_at: str | None = None) -> None:
        self.state = state
        self.fail_at = fail_at
        self.target = "electron_density"
        self.commits = 0
        self.rollbacks = 0
        self.transactions: list[_Transaction] = []

    def __enter__(self) -> _Graph:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def session(self) -> _Session:
        return _Session(self)


def _state(
    *,
    old_stage: str = "accepted",
    old_validation: str = "quarantined",
    target: str = "electron_density",
    target_unit: str = "m^-3",
    dd_unit: str = "m^-3",
    old_extra: dict[str, Any] | None = None,
    backing_id: str = "core_profiles/profiles_1d/electrons/density",
) -> _State:
    old = "invalid_duplicate"
    source_id = "dd:" + backing_id
    return _State(
        nodes={
            old: _node(
                old,
                stage=old_stage,
                validation=old_validation,
                paths=["dd:" + backing_id],
                **(old_extra or {}),
            ),
            target: _node(
                target,
                stage="accepted",
                validation="valid",
                unit=target_unit,
                paths=[],
                relationship_units=[target_unit],
            ),
        },
        sources={
            source_id: {
                "properties": {
                    "id": source_id,
                    "source_type": "dd",
                    "source_id": backing_id,
                    "status": "composed",
                    "produced_sn_id": old,
                    "claim_token": None,
                    "claimed_at": None,
                },
                "bindings": [old],
                "backings": [backing_id],
            }
        },
        backings={
            backing_id: {
                "labels": ["IMASNode"],
                "properties": {"id": backing_id, "unit": dd_unit},
                "units": [dd_unit],
                "projections": [old],
            }
        },
    )


def _run(
    graph: _Graph,
    *,
    old: str = "invalid_duplicate",
    target: str = "electron_density",
    dry_run: bool = False,
    parseable: bool = True,
    use_west_manifest: bool = False,
) -> dict[str, Any]:
    graph.target = target
    west_paths = edit._fold_west_dd_paths() if use_west_manifest else frozenset()
    with (
        patch.object(edit, "GraphClient", return_value=graph),
        patch.object(edit, "_fold_west_dd_paths", return_value=west_paths),
        patch.object(
            edit,
            "_isn_round_trip_ok",
            return_value=(parseable, None if parseable else "strict parse failed"),
        ),
    ):
        return edit.supersede_into(old, target, dry_run=dry_run)


@pytest.mark.parametrize(
    "stage", ["pending", "drafted", "reviewed", "accepted", "exhausted"]
)
def test_fold_preserves_actual_predecessor_stage(stage: str) -> None:
    graph = _Graph(_state(old_stage=stage))
    result = _run(graph)
    assert result["ok"] is True
    assert result["old_prior_stage"] == stage
    assert graph.state.nodes["invalid_duplicate"]["superseded_from_stage"] == stage


def test_fold_migrates_scalars_edges_cache_and_lineage_exactly() -> None:
    state = _state()
    source = next(iter(state.sources.values()))
    backing = next(iter(state.backings.values()))
    source["bindings"].extend(["invalid_duplicate", "electron_density"])
    backing["projections"].extend(["invalid_duplicate", "electron_density"])
    backing["properties"]["standard_name_id"] = "invalid_duplicate"
    graph = _Graph(state)
    result = _run(graph)
    source = next(iter(graph.state.sources.values()))
    backing = next(iter(graph.state.backings.values()))
    assert source["bindings"] == ["electron_density"]
    assert source["properties"]["produced_sn_id"] == "electron_density"
    assert backing["projections"] == ["electron_density"]
    assert backing["properties"]["standard_name_id"] == "electron_density"
    assert graph.state.nodes["invalid_duplicate"]["source_paths"] == []
    assert graph.state.nodes["electron_density"]["source_paths"] == [
        "dd:core_profiles/profiles_1d/electrons/density"
    ]
    assert graph.state.refined_from == [("electron_density", "invalid_duplicate")]
    assert result["receipt_counts"] == {
        "sources": 1,
        "projections": 1,
        "lineage": 1,
        "changes": 1,
    }


def test_exact_preexisting_target_lineage_is_admitted_without_duplicate() -> None:
    state = _state()
    state.refined_from.append(("electron_density", "invalid_duplicate"))
    graph = _Graph(state)
    assert _run(graph, dry_run=True)["ok"] is True
    assert _run(graph)["ok"] is True
    assert graph.state.refined_from == [("electron_density", "invalid_duplicate")]


def test_other_duplicate_and_cyclic_lineage_are_refused() -> None:
    other = _Graph(_state())
    other.state.nodes["other_density"] = _node("other_density", stage="accepted")
    other.state.refined_from.append(("other_density", "invalid_duplicate"))
    assert "another successor" in _run(other)["reason"]

    duplicate = _Graph(_state())
    duplicate.state.refined_from.extend(
        [
            ("electron_density", "invalid_duplicate"),
            ("electron_density", "invalid_duplicate"),
        ]
    )
    assert "duplicate target successor" in _run(duplicate)["reason"]

    cycle = _Graph(_state())
    cycle.state.refined_from.append(("invalid_duplicate", "electron_density"))
    assert "cycle" in _run(cycle)["reason"]


def test_scalar_only_source_is_discovered_and_repaired() -> None:
    state = _state()
    source = next(iter(state.sources.values()))
    backing = next(iter(state.backings.values()))
    source["bindings"] = []
    backing["projections"] = []
    graph = _Graph(state)
    result = _run(graph)
    assert result["sources_carried"] == 1
    assert source is not next(iter(graph.state.sources.values()))
    assert next(iter(graph.state.sources.values()))["bindings"] == ["electron_density"]
    assert next(iter(graph.state.backings.values()))["projections"] == [
        "electron_density"
    ]


@pytest.mark.parametrize("competitor_stage", ["accepted", "approved", "refining"])
@pytest.mark.parametrize("channel", ["scalar", "binding", "projection"])
def test_live_competitor_is_a_write_free_refusal(
    competitor_stage: str, channel: str
) -> None:
    state = _state()
    state.nodes["competing_density"] = _node(
        "competing_density", stage=competitor_stage
    )
    source = next(iter(state.sources.values()))
    backing = next(iter(state.backings.values()))
    if channel == "scalar":
        source["properties"]["produced_sn_id"] = "competing_density"
        source["bindings"] = []
    elif channel == "binding":
        source["bindings"].append("competing_density")
    else:
        backing["projections"].append("competing_density")
    graph = _Graph(state)
    before = copy.deepcopy(graph.state)
    result = _run(graph)
    assert result["ok"] is False
    assert "third live" in result["reason"]
    assert graph.state == before
    assert graph.transactions[0].write_markers == []


def test_third_historical_binding_and_projection_are_preserved() -> None:
    state = _state()
    state.nodes["retired_density"] = _node("retired_density", stage="superseded")
    next(iter(state.sources.values()))["bindings"].append("retired_density")
    next(iter(state.backings.values()))["projections"].append("retired_density")
    graph = _Graph(state)
    assert _run(graph)["ok"] is True
    assert next(iter(graph.state.sources.values()))["bindings"] == [
        "retired_density",
        "electron_density",
    ]
    assert next(iter(graph.state.backings.values()))["projections"] == [
        "retired_density",
        "electron_density",
    ]


def test_backing_owner_ambiguity_is_refused() -> None:
    state = _state()
    source_id = next(iter(state.sources))
    state.sources[source_id]["backings"].append(next(iter(state.backings)))
    assert "ambiguous backing cardinality" in _run(_Graph(state))["reason"]

    state = _state()
    backing_id = next(iter(state.backings))
    state.sources["dd:second"] = {
        "properties": {
            "id": "dd:second",
            "source_type": "dd",
            "source_id": backing_id,
            "status": "composed",
            "produced_sn_id": "electron_density",
            "claim_token": None,
            "claimed_at": None,
        },
        "bindings": ["electron_density"],
        "backings": [backing_id],
    }
    assert "ambiguous owner cardinality" in _run(_Graph(state))["reason"]


def test_scalar_and_relationship_unit_drift_is_refused() -> None:
    target = _state()
    target.nodes["electron_density"]["relationship_units"] = ["K"]
    assert "target" in _run(_Graph(target))["reason"]
    assert "scalar unit disagrees" in _run(_Graph(target))["reason"]

    backing = _state()
    next(iter(backing.backings.values()))["units"] = ["K"]
    result = _run(_Graph(backing))
    assert "backing" in result["reason"]
    assert "scalar unit disagrees" in result["reason"]


def test_attachment_mismatch_is_a_write_free_refusal() -> None:
    graph = _Graph(_state(target="change_in_electron_density"))
    before = copy.deepcopy(graph.state)
    result = _run(graph, target="change_in_electron_density")
    assert result["ok"] is False
    assert "tense mismatch" in result["reason"]
    assert graph.state == before


def test_exact_west_manifest_membership_fails_closed() -> None:
    west_path = sorted(edit._fold_west_dd_paths())[0]
    graph = _Graph(_state(backing_id=west_path))
    result = _run(graph, use_west_manifest=True)
    assert result["ok"] is False
    assert "WEST task manifest" in result["reason"]

    same_prefix_not_member = west_path.rsplit("/", 1)[0] + "/not_in_manifest"
    assert same_prefix_not_member not in edit._fold_west_dd_paths()
    assert (
        _run(_Graph(_state(backing_id=same_prefix_not_member)), use_west_manifest=True)[
            "ok"
        ]
        is True
    )


def test_receipt_is_full_typed_and_preserves_audit_subgraphs() -> None:
    state = _state(old_extra={"edit_status": "open", "edit_reason": "dedupe"})
    state.reviews["review:scalar-only"] = {
        "id": "review:scalar-only",
        "standard_name_id": "invalid_duplicate",
        "score": 0.7,
    }
    state.reviews["review:linked"] = {
        "id": "review:linked",
        "standard_name_id": "invalid_duplicate",
        "score": 0.8,
    }
    state.review_links.append(("invalid_duplicate", "review:linked"))
    state.revisions["revision:one"] = {"id": "revision:one", "documentation": "old"}
    state.revision_links.append(("invalid_duplicate", "revision:one"))
    state.changes["change:existing"] = {
        "id": "change:existing",
        "operation": "reclassify_domain",
        "reason": "kept",
    }
    state.change_links.append(("invalid_duplicate", "change:existing"))
    before_reviews = copy.deepcopy(state.reviews)
    before_revisions = copy.deepcopy(state.revisions)
    graph = _Graph(state)
    result = _run(graph)
    event = graph.state.changes[result["change_id"]]
    receipt = json.loads(event["reason"])
    assert receipt["receipt_type"] == edit._FOLD_RECEIPT_TYPE
    assert receipt["schema_version"] == edit._FOLD_RECEIPT_SCHEMA
    assert receipt["run_id"] == event["run_id"] == result["run_id"]
    assert receipt["before"]["reviews"]
    assert receipt["before"]["revisions"]
    assert receipt["before"]["changes"]
    assert receipt["expected_after"]["names"]["old"]["edit_status"] == "applied"
    assert graph.state.reviews == before_reviews
    assert graph.state.revisions == before_revisions
    assert graph.state.changes["change:existing"]["reason"] == "kept"
    assert (
        graph.state.change_links.count(("invalid_duplicate", result["change_id"])) == 1
    )
    assert (
        graph.state.change_links.count(("electron_density", result["change_id"])) == 1
    )


def test_dry_run_has_exact_plan_and_no_write() -> None:
    graph = _Graph(_state())
    before = copy.deepcopy(graph.state)
    result = _run(graph, dry_run=True)
    assert result["mutation_plan"]["source_ids"] == [
        "dd:core_profiles/profiles_1d/electrons/density"
    ]
    assert result["mutation_plan"]["backing_ids"] == [
        "core_profiles/profiles_1d/electrons/density"
    ]
    assert graph.state == before
    assert graph.commits == 0
    assert graph.transactions[0].write_markers == []


@pytest.mark.parametrize("failure", ["race", "event", "partial_source"])
def test_transaction_failure_rolls_back_everything(failure: str) -> None:
    graph = _Graph(_state(), fail_at=failure)
    before = copy.deepcopy(graph.state)
    match = {
        "race": "changed after preflight",
        "event": "event failure",
        "partial_source": "partial source migration",
    }[failure]
    with pytest.raises(RuntimeError, match=match):
        _run(graph)
    assert graph.state == before
    assert graph.commits == 0
    assert graph.rollbacks == 1


@pytest.mark.parametrize("failure", ["competitor_race", "unit_race"])
def test_locked_snapshot_detects_competitor_and_unit_races(failure: str) -> None:
    state = _state()
    state.nodes["retired_density"] = _node("retired_density", stage="superseded")
    next(iter(state.sources.values()))["bindings"].append("retired_density")
    graph = _Graph(state, fail_at=failure)
    before = copy.deepcopy(graph.state)
    with pytest.raises(RuntimeError, match="changed after preflight"):
        _run(graph)
    assert graph.state == before
    assert graph.commits == 0
    assert graph.rollbacks == 1


def test_exact_second_run_is_noop_and_all_partial_drift_is_refused() -> None:
    graph = _Graph(_state())
    first = _run(graph)
    committed = copy.deepcopy(graph.state)
    second = _run(graph)
    assert first["ok"] and second["ok"]
    assert second["already_superseded"] is True
    assert graph.commits == 1
    assert graph.state == committed
    assert graph.transactions[-1].write_markers == []

    drifts = []
    stale_cache = copy.deepcopy(committed)
    stale_cache.nodes["electron_density"]["source_paths"] = ["stale"]
    drifts.append(stale_cache)
    third_target = copy.deepcopy(committed)
    third_target.nodes["third"] = _node("third", stage="superseded")
    next(iter(third_target.sources.values()))["bindings"].append("third")
    drifts.append(third_target)
    third_projection = copy.deepcopy(committed)
    third_projection.nodes["third"] = _node("third", stage="superseded")
    next(iter(third_projection.backings.values()))["projections"].append("third")
    drifts.append(third_projection)
    changed_backing = copy.deepcopy(committed)
    backing = next(iter(changed_backing.backings.values()))
    backing["properties"]["id"] = "changed/path"
    drifts.append(changed_backing)
    duplicate_binding = copy.deepcopy(committed)
    next(iter(duplicate_binding.sources.values()))["bindings"].append(
        "electron_density"
    )
    drifts.append(duplicate_binding)
    partial_links = copy.deepcopy(committed)
    partial_links.change_links.remove(("electron_density", first["change_id"]))
    drifts.append(partial_links)

    for drifted in drifts:
        result = _run(_Graph(drifted))
        assert result["ok"] is False
        assert "drift" in result["reason"] or "ambiguous" in result["reason"]


def test_guards_target_claim_parse_and_stage_without_writes() -> None:
    reviewed = _Graph(_state())
    reviewed.state.nodes["electron_density"]["name_stage"] = "reviewed"
    assert "not 'accepted'" in _run(reviewed)["reason"]
    invalid = _Graph(_state())
    invalid.state.nodes["electron_density"]["validation_status"] = "quarantined"
    assert "not 'valid'" in _run(invalid)["reason"]
    claimed = _Graph(_state())
    claimed.state.nodes["invalid_duplicate"]["claim_token"] = "busy"
    assert "actively claimed" in _run(claimed)["reason"]
    unparsable = _Graph(_state())
    assert "strict ISN" in _run(unparsable, parseable=False)["reason"]


def test_self_missing_and_unrelated_state_are_safe() -> None:
    graph = _Graph(_state())
    unrelated = copy.deepcopy(graph.state.unrelated)
    assert "same" in _run(graph, old="electron_density")["reason"]
    assert "not found" in _run(_Graph(_state()), old="missing")["reason"]
    assert _run(graph)["ok"] is True
    assert graph.state.unrelated == unrelated


def test_production_queries_use_six_transaction_boundaries_without_temp_props() -> None:
    queries = [
        edit._FOLD_SNAPSHOT_QUERY,
        edit._FOLD_EVENT_QUERY,
        edit._FOLD_LOCK_QUERY,
        edit._FOLD_SOURCE_MUTATION_QUERY,
        edit._FOLD_NAME_MUTATION_QUERY,
        edit._FOLD_POSTFLIGHT_QUERY,
    ]
    assert len(queries) == 6
    assert all("_atomic_fold_lock" not in query for query in queries)
    assert "SET participant.id = participant.id" in edit._FOLD_LOCK_QUERY
