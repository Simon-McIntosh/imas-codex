"""Signed structural release admission across live and superseded names.

The release removes exactly one ``HAS_PARENT`` edge and creates nothing. These
tests run the admission, the delete and the postcondition against a faked
graph, so the lifecycle table is exercised without a database.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.graph.models import RepairMutationKind
from imas_codex.standard_names import signed_manifest as operator
from imas_codex.standard_names.signed_manifest import SignedManifestConflict

CHILD = "surface_area_of_flux_surface"
PARENT = "area_of_flux_surface"
EDGE = "relationship-child-parent"
EDGE_PROPERTIES = {"operator_kind": "unary", "operator": "surface"}

ACCEPTED = {
    "name_stage": "accepted",
    "validation_status": "valid",
    "status": "draft",
}
SUPERSEDED = {
    "name_stage": "superseded",
    "validation_status": "valid",
    "status": "superseded",
}
DEPRECATED = {
    "name_stage": "accepted",
    "validation_status": "valid",
    "status": "deprecated",
}
INVALID = {
    "name_stage": "accepted",
    "validation_status": "invalid",
    "status": "draft",
}

LIVE_PRODUCER = {
    "source_element_id": "source-1",
    "source_id": "dd:equilibrium/grid/surface",
    "source_properties": {"status": "attached"},
    "binding_element_id": "binding-1",
    "binding_properties": {"authority": "dd"},
}
STALE_PRODUCER = {
    **LIVE_PRODUCER,
    "source_properties": {"status": "stale"},
}


class _Graph:
    """Answers the statements a structural release issues."""

    def __init__(
        self,
        child: dict[str, Any],
        parent: dict[str, Any],
        producers: list[dict[str, Any]] | None = None,
        extra_parents: tuple[str, ...] = (),
    ) -> None:
        self.nodes = {
            CHILD: {"element_id": "node-child", "properties": dict(child)},
            PARENT: {"element_id": "node-parent", "properties": dict(parent)},
        }
        for index, name in enumerate(extra_parents):
            self.nodes[name] = {
                "element_id": f"node-extra-{index}",
                "properties": dict(ACCEPTED),
            }
        self.edges: dict[str, dict[str, Any]] = {
            EDGE: {"start": CHILD, "end": PARENT, "properties": dict(EDGE_PROPERTIES)}
        }
        for index, name in enumerate(extra_parents):
            self.edges[f"relationship-extra-{index}"] = {
                "start": CHILD,
                "end": name,
                "properties": {"operator": "extra"},
            }
        self.producers = [dict(item) for item in producers or []]
        self.deleted: list[str] = []

    def _parents(self) -> list[dict[str, Any]]:
        return [
            {
                "relationship_id": edge_id,
                "relationship_properties": edge["properties"],
                "parent_element_id": self.nodes[edge["end"]]["element_id"],
                "parent_id": edge["end"],
            }
            for edge_id, edge in self.edges.items()
            if edge["start"] == CHILD
        ]

    def query(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if "WHERE node.id = $participant_id" in cypher:
            node = self.nodes.get(params["participant_id"])
            if node is None:
                return []
            return [
                {
                    "element_id": node["element_id"],
                    "labels": ["StandardName"],
                    "properties": dict(node["properties"]),
                    "property_fingerprint": [],
                }
            ]
        if "WHERE elementId(relationship) = $participant_id" in cypher:
            edge = self.edges.get(params["participant_id"])
            if edge is None:
                return []
            return [
                {
                    "element_id": params["participant_id"],
                    "relationship_type": "HAS_PARENT",
                    "properties": dict(edge["properties"]),
                    "start_element_id": self.nodes[edge["start"]]["element_id"],
                    "start_labels": ["StandardName"],
                    "start_id": edge["start"],
                    "start_status": None,
                    "end_element_id": self.nodes[edge["end"]]["element_id"],
                    "end_labels": ["StandardName"],
                    "end_id": edge["end"],
                }
            ]
        if "AS producers" in cypher and "$child_element_id" in cypher:
            return [
                {
                    "child_properties": dict(self.nodes[CHILD]["properties"]),
                    "parents": self._parents(),
                    "producers": [dict(item) for item in self.producers],
                }
            ]
        if "AS producers" in cypher and "{id: $child_id}" in cypher:
            return [
                {
                    "child_properties": dict(self.nodes[CHILD]["properties"]),
                    "old_parent_properties": dict(self.nodes[PARENT]["properties"]),
                    "parents": [
                        {
                            "relationship_properties": item["relationship_properties"],
                            "parent_id": item["parent_id"],
                        }
                        for item in self._parents()
                    ],
                    "producers": [dict(item) for item in self.producers],
                }
            ]
        if "DELETE prior" in cypher:
            edge = self.edges.get(params["relationship_element_id"])
            if (
                edge is None
                or len(self._parents()) != 1
                or edge["properties"] != params["relationship_properties"]
            ):
                return []
            del self.edges[params["relationship_element_id"]]
            self.deleted.append(params["relationship_element_id"])
            return [{"child_id": CHILD}]
        raise AssertionError(f"unexpected statement: {cypher}")


def _row() -> operator._LoadedRow:
    return operator._LoadedRow(
        id=CHILD,
        identity={"id": CHILD, "kind": "standard_name", "target_id": CHILD},
        participants=(
            {"id": CHILD, "kind": "node", "graph_label": "StandardName"},
            {"id": PARENT, "kind": "node", "graph_label": "StandardName"},
            {"id": EDGE, "kind": "relationship", "graph_label": "HAS_PARENT"},
        ),
        mutations=(
            {
                "id": "release-parent-edge",
                "order": 0,
                "kind": RepairMutationKind.recompute_projection.value,
                "participant_id": EDGE,
                "arguments": {
                    "relationship_type": "HAS_PARENT",
                    "start_id": CHILD,
                    "old_end_id": PARENT,
                    "new_end_id": None,
                    "properties": dict(EDGE_PROPERTIES),
                },
            },
        ),
        guards=(),
        orphan_policy="refuse",
    )


def _action(graph: _Graph) -> dict[str, Any]:
    row = _row()
    snapshots = {}
    for participant in row.participants:
        state = operator._participant_snapshot(graph, dict(participant))
        assert state is not None
        snapshots[str(participant["id"])] = state[0]
    return {"row": row, "participant_snapshots": snapshots, "participant_digests": []}


def _admit(graph: _Graph) -> tuple[str | None, dict[str, Any]]:
    action = _action(graph)
    return operator._structural_release_refusal(graph, action), action


def _release(graph: _Graph) -> dict[str, Any]:
    refusal, action = _admit(graph)
    assert refusal is None
    assert operator._apply_structural_release(graph, action) == 1
    assert graph.deleted == [EDGE]
    assert not graph._parents()
    operator._verify_structural_release_postcondition(graph, action["row"], action)
    return action


def test_accepted_child_with_accepted_parent_is_released() -> None:
    graph = _Graph(ACCEPTED, ACCEPTED, [LIVE_PRODUCER])

    action = _release(graph)

    assert action["structural_release_retirement"] == {
        "child": False,
        "old_parent": False,
    }


@pytest.mark.parametrize(
    "parent",
    [
        SUPERSEDED,
        {"name_stage": "superseded", "validation_status": "valid", "status": "draft"},
        {
            "name_stage": "accepted",
            "validation_status": "valid",
            "status": "superseded",
        },
    ],
    ids=["stage-and-status", "stage-only", "status-only"],
)
def test_accepted_child_with_superseded_parent_is_released(
    parent: dict[str, Any],
) -> None:
    graph = _Graph(ACCEPTED, parent, [LIVE_PRODUCER])

    action = _release(graph)

    assert action["structural_release_retirement"] == {
        "child": False,
        "old_parent": True,
    }
    assert graph.nodes[CHILD]["properties"] == ACCEPTED
    assert graph.nodes[PARENT]["properties"] == parent
    assert graph.producers == [LIVE_PRODUCER]


@pytest.mark.parametrize(
    "producers",
    [[], [STALE_PRODUCER], [LIVE_PRODUCER]],
    ids=["unbound", "stale-producer", "live-producer"],
)
def test_superseded_child_with_superseded_parent_is_released(
    producers: list[dict[str, Any]],
) -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED, producers)

    action = _release(graph)

    assert action["structural_release_retirement"] == {
        "child": True,
        "old_parent": True,
    }
    assert graph.nodes[CHILD]["properties"] == SUPERSEDED
    assert graph.nodes[PARENT]["properties"] == SUPERSEDED
    assert graph.producers == producers


def test_superseded_child_needs_no_valid_validation_status() -> None:
    child = {**SUPERSEDED, "validation_status": "invalid"}
    graph = _Graph(child, SUPERSEDED)

    _release(graph)


def test_superseded_child_with_accepted_parent_is_refused_by_name() -> None:
    graph = _Graph(SUPERSEDED, ACCEPTED, [LIVE_PRODUCER])

    refusal, _ = _admit(graph)

    assert refusal == (
        "signed structural release superseded child requires a superseded old parent"
    )
    assert graph.deleted == []
    assert EDGE in graph.edges


def test_accepted_child_still_requires_producer_authority_under_superseded_parent() -> (
    None
):
    for producers in ([], [STALE_PRODUCER]):
        graph = _Graph(ACCEPTED, SUPERSEDED, producers)

        refusal, _ = _admit(graph)

        assert refusal == "signed structural release child has no live producing source"
        assert EDGE in graph.edges


@pytest.mark.parametrize(
    ("child", "parent", "producers", "message"),
    [
        (
            ACCEPTED,
            DEPRECATED,
            [LIVE_PRODUCER],
            "signed structural release old parent is deprecated",
        ),
        (
            SUPERSEDED,
            DEPRECATED,
            [],
            "signed structural release old parent is deprecated",
        ),
        (
            DEPRECATED,
            ACCEPTED,
            [LIVE_PRODUCER],
            "signed structural release child is deprecated",
        ),
        (
            DEPRECATED,
            SUPERSEDED,
            [LIVE_PRODUCER],
            "signed structural release child is deprecated",
        ),
        (
            INVALID,
            ACCEPTED,
            [LIVE_PRODUCER],
            "signed structural release accepted child is not valid",
        ),
        (
            INVALID,
            SUPERSEDED,
            [LIVE_PRODUCER],
            "signed structural release accepted child is not valid",
        ),
        (
            {**ACCEPTED, "name_stage": "exhausted"},
            SUPERSEDED,
            [LIVE_PRODUCER],
            "signed structural release child is neither accepted nor superseded",
        ),
        (
            ACCEPTED,
            {**ACCEPTED, "name_stage": "exhausted"},
            [LIVE_PRODUCER],
            "signed structural release old parent is neither accepted nor superseded",
        ),
    ],
    ids=[
        "deprecated-parent-accepted-child",
        "deprecated-parent-superseded-child",
        "deprecated-child-accepted-parent",
        "deprecated-child-superseded-parent",
        "invalid-child-accepted-parent",
        "invalid-child-superseded-parent",
        "exhausted-child",
        "exhausted-parent",
    ],
)
def test_other_lifecycle_combinations_are_refused_naming_the_condition(
    child: dict[str, Any],
    parent: dict[str, Any],
    producers: list[dict[str, Any]],
    message: str,
) -> None:
    graph = _Graph(child, parent, producers)

    refusal, _ = _admit(graph)

    assert refusal == message
    assert EDGE in graph.edges


def test_superseded_child_keeps_the_one_parent_closure() -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED, extra_parents=("other_parent",))

    refusal, _ = _admit(graph)

    assert refusal == (
        "signed structural release closure does not match exact incumbent parent"
    )


def test_superseded_child_keeps_the_exact_edge_properties() -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED)
    action = _action(graph)
    graph.edges[EDGE]["properties"] = {"operator": "drifted"}
    action["participant_snapshots"][EDGE]["properties"] = {"operator": "drifted"}

    refusal = operator._structural_release_refusal(graph, action)

    assert refusal == (
        "signed structural release closure does not match exact incumbent parent"
    )


def test_release_records_the_child_authority_digest() -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED, [LIVE_PRODUCER])

    _, action = _admit(graph)

    assert [item["participant_id"] for item in action["participant_digests"]] == [
        "structural-release-child-authority"
    ]
    assert action["structural_release_child_state"] == {
        "child_properties": SUPERSEDED,
        "producers": [LIVE_PRODUCER],
    }


def test_postcondition_refuses_an_edge_that_was_not_removed() -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED)
    _, action = _admit(graph)

    with pytest.raises(SignedManifestConflict, match="parentless postcondition"):
        operator._verify_structural_release_postcondition(graph, action["row"], action)


def test_postcondition_refuses_a_parent_that_stopped_being_superseded() -> None:
    graph = _Graph(ACCEPTED, SUPERSEDED, [LIVE_PRODUCER])
    _, action = _admit(graph)
    operator._apply_structural_release(graph, action)
    graph.nodes[PARENT]["properties"] = dict(ACCEPTED)

    with pytest.raises(SignedManifestConflict, match="supersession"):
        operator._verify_structural_release_postcondition(graph, action["row"], action)


def test_postcondition_refuses_a_child_that_stopped_being_superseded() -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED)
    _, action = _admit(graph)
    operator._apply_structural_release(graph, action)
    graph.nodes[CHILD]["properties"] = dict(ACCEPTED)

    with pytest.raises(SignedManifestConflict, match="child lifecycle"):
        operator._verify_structural_release_postcondition(graph, action["row"], action)


def test_postcondition_refuses_changed_producers_of_a_released_child() -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED, [LIVE_PRODUCER])
    _, action = _admit(graph)
    operator._apply_structural_release(graph, action)
    graph.producers = []

    with pytest.raises(SignedManifestConflict, match="producing-source authority"):
        operator._verify_structural_release_postcondition(graph, action["row"], action)


def test_replayed_release_needs_only_the_parentless_postcondition() -> None:
    graph = _Graph(SUPERSEDED, SUPERSEDED)
    action = _release(graph)

    operator._verify_structural_release_postcondition(graph, action["row"], None)
