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
    refined_from: set[tuple[str, str]] = field(default_factory=set)
    changes: dict[str, dict[str, Any]] = field(default_factory=dict)
    change_links: set[tuple[str, str]] = field(default_factory=set)
    unrelated: dict[str, Any] = field(default_factory=lambda: {"sentinel": [1, 2, 3]})


class _Transaction:
    """Copy-on-write graph transaction with injectable failure boundaries."""

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

    def _relationships(self, old: str, target: str) -> list[dict[str, Any]]:
        relationships: list[dict[str, Any]] = []
        old_element_id = self._element_id("name", old)
        target_element_id = self._element_id("name", target)

        def add(kind: str, key: str, start: str, end: str) -> None:
            if start not in {old_element_id, target_element_id} and end not in {
                old_element_id,
                target_element_id,
            }:
                return
            relationships.append(
                {
                    "element_id": f"rel:{kind}:{key}",
                    "type": kind,
                    "start_element_id": start,
                    "end_element_id": end,
                    "other_element_id": (
                        end if start in {old_element_id, target_element_id} else start
                    ),
                    "properties": {},
                }
            )

        for source_id, source in self.state.sources.items():
            for bound in source["bound"]:
                add(
                    "PRODUCED_NAME",
                    f"{source_id}:{bound}",
                    self._element_id("source", source_id),
                    self._element_id("name", bound),
                )
        for backing_id, backing in self.state.backings.items():
            for projected in backing["projected"]:
                add(
                    "HAS_STANDARD_NAME",
                    f"{backing_id}:{projected}",
                    self._element_id("backing", backing_id),
                    self._element_id("name", projected),
                )
        for successor, predecessor in self.state.refined_from:
            add(
                "REFINED_FROM",
                f"{successor}:{predecessor}",
                self._element_id("name", successor),
                self._element_id("name", predecessor),
            )
        for owner, change_id in self.state.change_links:
            add(
                "HAS_INTERNAL_CHANGE",
                f"{owner}:{change_id}",
                self._element_id("name", owner),
                self._element_id("change", change_id),
            )
        return sorted(relationships, key=lambda value: value["element_id"])

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

    def _source_row(self, source_id: str, source: dict[str, Any]) -> dict[str, Any]:
        backings = []
        for backing_id in source.get("backings", []):
            backing = self.state.backings[backing_id]
            backings.append(
                {
                    "id": backing_id,
                    "element_id": self._element_id("backing", backing_id),
                    "labels": list(backing["labels"]),
                    "properties": copy.deepcopy(backing["properties"]),
                    "units": list(backing.get("units", [])),
                    "projected": sorted(
                        [
                            {
                                "id": name,
                                "stage": self.state.nodes[name]["name_stage"],
                            }
                            for name in backing["projected"]
                        ],
                        key=lambda value: value["id"],
                    ),
                }
            )
        return {
            "id": source_id,
            "element_id": self._element_id("source", source_id),
            "properties": copy.deepcopy(source["properties"]),
            "bound_ids": sorted(source["bound"]),
            "live_targets": sorted(
                name
                for name in source["bound"]
                if self.state.nodes[name]["name_stage"] in edit._FOLD_LIVE_STAGES
            ),
            "backings": sorted(backings, key=lambda value: value["id"]),
        }

    def _snapshot(self, old: str, target: str) -> list[dict[str, Any]]:
        self.snapshot_count += 1
        if old not in self.state.nodes or target not in self.state.nodes:
            return []
        sources = [
            self._source_row(source_id, source)
            for source_id, source in self.state.sources.items()
            if source["bound"] & {old, target}
        ]
        fold_events = [
            copy.deepcopy(change)
            for change_id, change in self.state.changes.items()
            if change.get("operation") == "fold_identity"
            and change.get("from_name") == old
            and change.get("to_name") == target
            and any(
                (owner, change_id) in self.state.change_links for owner in (old, target)
            )
        ]
        return [
            {
                "old_element_id": self._element_id("name", old),
                "target_element_id": self._element_id("name", target),
                "old_properties": copy.deepcopy(self.state.nodes[old]),
                "target_properties": copy.deepcopy(self.state.nodes[target]),
                "cycle": self._descends(old, target),
                "sources": sorted(sources, key=lambda value: value["id"]),
                "relationships": self._relationships(old, target),
                "fold_events": sorted(fold_events, key=lambda value: value["id"]),
                "target_units": list(
                    self.state.nodes[target].get("relationship_units", [])
                ),
            }
        ]

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "ATOMIC_FOLD_SNAPSHOT" in cypher:
            return self._snapshot(params["old_id"], params["into_id"])
        if "ATOMIC_FOLD_LOCK" in cypher:
            if self.graph.fail_at == "race":
                self.state.nodes[self.graph.target]["documentation"] = "concurrent"
            return [{"locked": len(params["element_ids"])}]
        if "RETURN source_id," in cypher and "already_bound" in cypher:
            target = params["sn_id"]
            existing = sorted(
                {
                    backing_id
                    for source in self.state.sources.values()
                    if target in source["bound"]
                    for backing_id in source.get("backings", [])
                    if "IMASNode" in self.state.backings[backing_id]["labels"]
                }
            )
            rows = []
            for source_id in params["source_ids"]:
                source = self.state.sources[source_id]
                backing = self.state.backings[source["backings"][0]]
                is_dd = "IMASNode" in backing["labels"]
                rows.append(
                    {
                        "source_id": source_id,
                        "source_type": source["properties"]["source_type"],
                        "dd_path": backing["properties"]["id"] if is_dd else None,
                        "dd_unit": (
                            backing["units"][0]
                            if backing.get("units")
                            else backing["properties"].get("unit")
                        ),
                        "sn_unit": self.state.nodes[target]["unit"],
                        "already_bound": target in source["bound"],
                        "existing_dd_paths": existing,
                        "name_stage": self.state.nodes[target]["name_stage"],
                    }
                )
            return rows
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
                "changed_at": params["changed_at"],
                "internal": True,
            }
            self.state.changes[change["id"]] = change
            self.state.change_links.add((params["old_id"], change["id"]))
            self.state.change_links.add((params["into_id"], change["id"]))
            return [{"change_id": change["id"]}]
        if "ATOMIC_FOLD_MOVE_SOURCES" in cypher:
            self.write_markers.append("sources")
            moved_backings: set[str] = set()
            for index, expected in enumerate(params["sources"]):
                source = self.state.sources[expected["id"]]
                if source["properties"] != expected["properties"]:
                    continue
                source["bound"].discard(params["old_id"])
                source["bound"].add(params["into_id"])
                source["properties"]["produced_sn_id"] = params["into_id"]
                for backing_id in source["backings"]:
                    backing = self.state.backings[backing_id]
                    backing["projected"].discard(params["old_id"])
                    backing["projected"].add(params["into_id"])
                    moved_backings.add(backing_id)
                if self.graph.fail_at == "partial_source" and index == 0:
                    raise RuntimeError("injected partial source migration")
            return [
                {
                    "sources_moved": len(params["sources"]),
                    "projections_moved": len(moved_backings),
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
            self.state.refined_from.add((params["into_id"], params["old_id"]))
            return [
                {
                    "old_stage": "superseded",
                    "predecessor_stage": params["predecessor_stage"],
                }
            ]
        if "ATOMIC_FOLD_POSTFLIGHT" in cypher:
            old = self.state.nodes[params["old_id"]]
            target = self.state.nodes[params["into_id"]]
            sources = [self.state.sources[value] for value in params["source_ids"]]
            correct_sources = sum(
                1
                for source in sources
                if source["properties"]["produced_sn_id"] == params["into_id"]
                and source["bound"] == {params["into_id"]}
            )
            correct_projections = sum(
                1
                for source in sources
                if all(
                    params["into_id"] in self.state.backings[backing]["projected"]
                    and params["old_id"]
                    not in self.state.backings[backing]["projected"]
                    for backing in source["backings"]
                )
            )
            return [
                {
                    "old_stage": old["name_stage"],
                    "predecessor_stage": old.get("superseded_from_stage"),
                    "old_claim_token": old.get("claim_token"),
                    "old_claimed_at": old.get("claimed_at"),
                    "old_paths": old.get("source_paths"),
                    "old_edit_status": old.get("edit_status"),
                    "target_stage": target["name_stage"],
                    "target_validation": target["validation_status"],
                    "target_claim_token": target.get("claim_token"),
                    "target_claimed_at": target.get("claimed_at"),
                    "target_paths": target.get("source_paths"),
                    "lineage_count": int(
                        (params["into_id"], params["old_id"]) in self.state.refined_from
                    ),
                    "event_count": int(params["change_id"] in self.state.changes),
                    "source_count": len(sources),
                    "correct_sources": correct_sources,
                    "correct_projections": correct_projections,
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
) -> _State:
    old = "invalid_duplicate"
    source_id = "dd:core_profiles/profiles_1d/electrons/density"
    backing_id = "core_profiles/profiles_1d/electrons/density"
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
                "bound": {old},
                "backings": [backing_id],
            }
        },
        backings={
            backing_id: {
                "labels": ["IMASNode"],
                "properties": {"id": backing_id, "unit": dd_unit},
                "units": [dd_unit],
                "projected": {old},
            }
        },
    )


def _run(
    graph: _Graph,
    *,
    old: str = "invalid_duplicate",
    target: str = "electron_density",
    dry_run: bool = False,
    include_west: bool = False,
    parseable: bool = True,
) -> dict[str, Any]:
    graph.target = target
    with (
        patch.object(edit, "GraphClient", return_value=graph),
        patch.object(
            edit,
            "_isn_round_trip_ok",
            return_value=(parseable, None if parseable else "strict parse failed"),
        ),
    ):
        return edit.supersede_into(
            old,
            target,
            dry_run=dry_run,
            include_west=include_west,
        )


@pytest.mark.parametrize(
    "stage", ["pending", "drafted", "reviewed", "accepted", "exhausted"]
)
def test_fold_preserves_actual_predecessor_stage(stage: str) -> None:
    graph = _Graph(_state(old_stage=stage))
    result = _run(graph)
    assert result["ok"] is True
    assert result["old_prior_stage"] == stage
    assert graph.state.nodes["invalid_duplicate"]["superseded_from_stage"] == stage
    assert graph.state.nodes["invalid_duplicate"]["name_stage"] == "superseded"


def test_pending_quarantined_and_accepted_fold_through_same_transaction() -> None:
    pending = _Graph(_state(old_stage="pending", old_validation="quarantined"))
    accepted = _Graph(_state(old_stage="accepted", old_validation="valid"))
    assert _run(pending)["ok"]
    assert _run(accepted)["ok"]
    assert pending.commits == accepted.commits == 1
    assert pending.transactions[0].write_markers == ["event", "sources", "names"]


def test_fold_migrates_sources_projections_scalars_cache_and_lineage() -> None:
    graph = _Graph(_state())
    result = _run(graph)
    source = next(iter(graph.state.sources.values()))
    backing = next(iter(graph.state.backings.values()))
    assert source["bound"] == {"electron_density"}
    assert source["properties"]["produced_sn_id"] == "electron_density"
    assert backing["projected"] == {"electron_density"}
    assert graph.state.nodes["invalid_duplicate"]["source_paths"] == []
    assert graph.state.nodes["electron_density"]["source_paths"] == [
        "dd:core_profiles/profiles_1d/electrons/density"
    ]
    assert ("electron_density", "invalid_duplicate") in graph.state.refined_from
    assert result["receipt_counts"] == {
        "sources": 1,
        "projections": 1,
        "lineage": 1,
        "changes": 1,
    }


def test_open_edit_resolves_only_on_folded_identity_with_mechanism_receipt() -> None:
    graph = _Graph(_state(old_extra={"edit_status": "open", "edit_reason": "dedupe"}))
    _run(graph)
    assert graph.state.nodes["invalid_duplicate"]["edit_status"] == "applied"
    assert graph.state.nodes["electron_density"].get("edit_status") is None
    change = next(iter(graph.state.changes.values()))
    receipt = json.loads(change["reason"])
    assert receipt["mechanism"] == edit._FOLD_REASON
    assert ("invalid_duplicate", change["id"]) in graph.state.change_links
    assert ("electron_density", change["id"]) in graph.state.change_links


def test_dry_run_has_exact_plan_and_zero_writes_or_commit() -> None:
    graph = _Graph(_state())
    before = copy.deepcopy(graph.state)
    result = _run(graph, dry_run=True)
    assert result["mutation_plan"]["source_ids"] == [
        "dd:core_profiles/profiles_1d/electrons/density"
    ]
    assert result["mutation_plan"]["predecessor_stage"] == "accepted"
    assert graph.state == before
    assert graph.commits == 0
    assert graph.transactions[0].write_markers == []


def test_concurrent_edit_after_snapshot_rolls_back_everything() -> None:
    graph = _Graph(_state(), fail_at="race")
    before = copy.deepcopy(graph.state)
    with pytest.raises(RuntimeError, match="changed after preflight"):
        _run(graph)
    assert graph.state == before
    assert graph.commits == 0
    assert graph.rollbacks == 1


def test_event_failure_rolls_back_without_tombstone_or_retarget() -> None:
    graph = _Graph(_state(), fail_at="event")
    before = copy.deepcopy(graph.state)
    with pytest.raises(RuntimeError, match="event failure"):
        _run(graph)
    assert graph.state == before
    assert graph.commits == 0


def test_partial_retarget_failure_rolls_back_event_and_source() -> None:
    graph = _Graph(_state(), fail_at="partial_source")
    before = copy.deepcopy(graph.state)
    with pytest.raises(RuntimeError, match="partial source migration"):
        _run(graph)
    assert graph.state == before
    assert graph.commits == 0


def test_unit_mismatch_is_a_write_free_refusal() -> None:
    graph = _Graph(_state(dd_unit="K"))
    before = copy.deepcopy(graph.state)
    result = _run(graph)
    assert result["ok"] is False
    assert "unit dimensionality mismatch" in result["reason"]
    assert graph.state == before
    assert graph.transactions[0].write_markers == []


def test_attachment_mismatch_is_a_write_free_refusal() -> None:
    graph = _Graph(_state(target="change_in_electron_density"))
    before = copy.deepcopy(graph.state)
    result = _run(graph, target="change_in_electron_density")
    assert result["ok"] is False
    assert "tense mismatch" in result["reason"]
    assert graph.state == before


def test_target_must_be_valid_accepted_and_strict_parseable() -> None:
    reviewed = _Graph(_state())
    reviewed.state.nodes["electron_density"]["name_stage"] = "reviewed"
    assert "not 'accepted'" in _run(reviewed)["reason"]
    invalid = _Graph(_state())
    invalid.state.nodes["electron_density"]["validation_status"] = "quarantined"
    assert "not 'valid'" in _run(invalid)["reason"]
    unparsable = _Graph(_state())
    assert "strict ISN" in _run(unparsable, parseable=False)["reason"]


def test_claim_and_third_live_target_refuse_ambiguity() -> None:
    claimed = _Graph(_state())
    claimed.state.nodes["invalid_duplicate"]["claim_token"] = "busy"
    assert "actively claimed" in _run(claimed)["reason"]

    ambiguous = _Graph(_state())
    ambiguous.state.nodes["other_density"] = _node(
        "other_density", stage="accepted", validation="valid"
    )
    source = next(iter(ambiguous.state.sources.values()))
    source["bound"].add("other_density")
    assert "multiple live targets" in _run(ambiguous)["reason"]


def test_west_signal_requires_explicit_authorization() -> None:
    state = _state()
    source_id = next(iter(state.sources))
    backing_id = next(iter(state.backings))
    state.sources[source_id]["properties"].update(
        {"id": "signals:west:density", "source_type": "signals"}
    )
    state.sources["signals:west:density"] = state.sources.pop(source_id)
    state.backings[backing_id]["labels"] = ["FacilitySignal"]
    state.backings[backing_id]["properties"]["facility_id"] = "west"
    graph = _Graph(state)
    assert "WEST" in _run(graph)["reason"]
    authorized = _Graph(copy.deepcopy(state))
    assert _run(authorized, include_west=True)["ok"]


def test_exact_second_run_is_write_free_and_drift_is_refused() -> None:
    graph = _Graph(_state())
    first = _run(graph)
    committed = copy.deepcopy(graph.state)
    second = _run(graph)
    assert first["ok"] and second["ok"]
    assert second["already_superseded"] is True
    assert second["old_prior_stage"] == "accepted"
    assert graph.commits == 1
    assert graph.state == committed
    assert graph.transactions[-1].write_markers == []

    graph.state.nodes["invalid_duplicate"]["source_paths"] = ["stale"]
    drift = _run(graph)
    assert drift["ok"] is False
    assert "drift" in drift["reason"]
    assert graph.commits == 1


def test_unrelated_state_is_invariant() -> None:
    graph = _Graph(_state())
    unrelated = copy.deepcopy(graph.state.unrelated)
    _run(graph)
    assert graph.state.unrelated == unrelated


def test_self_missing_and_target_cycle_are_write_free_refusals() -> None:
    graph = _Graph(_state())
    assert "same" in _run(graph, old="electron_density")["reason"]
    missing = _Graph(_state())
    assert "not found" in _run(missing, old="missing")["reason"]

    cycle = _Graph(_state())
    cycle.state.refined_from.add(("invalid_duplicate", "electron_density"))
    # The exact relationship snapshot makes a reverse lineage visible. The
    # operation must refuse before adding the closing edge.
    result = _run(cycle)
    assert result["ok"] is False
    assert "cycle" in result["reason"]

    successor = _Graph(_state())
    successor.state.nodes["other_density"] = _node(
        "other_density", stage="accepted", validation="valid"
    )
    successor.state.refined_from.add(("other_density", "invalid_duplicate"))
    result = _run(successor)
    assert result["ok"] is False
    assert "successor lineage" in result["reason"]
