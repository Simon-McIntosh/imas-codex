"""The sanctioned removal of one directed StandardName lineage relationship."""

from __future__ import annotations

from copy import deepcopy
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

_SUCCESSOR = "beta"
_PREDECESSOR = "normalized_toroidal_plasma_beta"
_ALTERNATE_SUCCESSOR = "normalized_toroidal_beta"
_REASON = "beta is live and did not supersede the normalized identity"


class _LineageGraph:
    """In-memory graph answering only the two lineage-removal statements."""

    def __init__(
        self,
        *,
        predecessor_stage: str = "superseded",
        alternate_successor: bool = True,
    ) -> None:
        self.names = {
            _SUCCESSOR: {"name_stage": "reviewed"},
            _PREDECESSOR: {"name_stage": predecessor_stage},
            _ALTERNATE_SUCCESSOR: {"name_stage": "accepted"},
        }
        self.edges = {(_SUCCESSOR, _PREDECESSOR)}
        if alternate_successor:
            self.edges.add((_ALTERNATE_SUCCESSOR, _PREDECESSOR))
        self.changes: list[dict[str, Any]] = []
        self.writes: list[str] = []

    def relationship_count(self) -> int:
        return len(self.edges)

    def shape(self) -> dict[str, Any]:
        return {
            "names": deepcopy(self.names),
            "edges": sorted(self.edges),
            "changes": deepcopy(self.changes),
        }

    def query(self, statement: str, **params: Any) -> list[dict[str, Any]]:
        from iter_codex.standard_names import edit

        if statement == edit._LINEAGE_REMOVAL_PREFLIGHT_QUERY:
            return [self._preflight(**params)]
        if statement == edit._LINEAGE_REMOVAL_QUERY:
            return self._remove(**params)
        raise AssertionError(f"unrecognized query: {statement[:100]!r}")

    def close(self) -> None:  # pragma: no cover - caller owns the borrowed graph
        raise AssertionError("the function must not close a borrowed graph")

    def _preflight(self, *, successor_id: str, predecessor_id: str) -> dict[str, Any]:
        directed = int((successor_id, predecessor_id) in self.edges)
        reverse = int((predecessor_id, successor_id) in self.edges)
        return {
            "successor_exists": successor_id in self.names,
            "predecessor_exists": predecessor_id in self.names,
            "predecessor_stage": self.names.get(predecessor_id, {}).get("name_stage"),
            "directed_edges": directed,
            "reverse_edges": reverse,
            "remaining_inbound": len(
                [
                    edge
                    for edge in self.edges
                    if edge[1] == predecessor_id and edge[0] != successor_id
                ]
            ),
            "remaining_outbound": len(
                [edge for edge in self.edges if edge[0] == predecessor_id]
            ),
        }

    def _remove(
        self,
        *,
        successor_id: str,
        predecessor_id: str,
        change_id: str,
        reason: str,
        changed_at: str,
    ) -> list[dict[str, Any]]:
        row = self._preflight(successor_id=successor_id, predecessor_id=predecessor_id)
        if row["directed_edges"] != 1:
            return []
        if row["predecessor_stage"] == "superseded" and row["remaining_inbound"] == 0:
            return []
        self.writes.append("remove_lineage")
        self.edges.remove((successor_id, predecessor_id))
        self.changes.append(
            {
                "id": change_id,
                "from_name": predecessor_id,
                "to_name": successor_id,
                "operation": "remove_refined_from_relationship",
                "reason": reason,
                "origin": "lineage_adjudication",
                "changed_at": changed_at,
            }
        )
        return [
            {
                "change_id": change_id,
                "remaining_inbound": row["remaining_inbound"],
                "remaining_outbound": row["remaining_outbound"],
            }
        ]


def test_removes_misdirected_beta_lineage_and_records_reason() -> None:
    """The motivating reverse edge is removed with its judgement retained."""
    from iter_codex.standard_names.edit import remove_refined_from_relationship

    graph = _LineageGraph()
    before_count = graph.relationship_count()

    result = remove_refined_from_relationship(
        _SUCCESSOR, _PREDECESSOR, reason=_REASON, gc=graph
    )

    assert result["ok"] is True
    assert graph.relationship_count() == before_count - 1
    assert (_SUCCESSOR, _PREDECESSOR) not in graph.edges
    assert (_ALTERNATE_SUCCESSOR, _PREDECESSOR) in graph.edges
    assert len(graph.changes) == 1
    change = graph.changes[0]
    assert change["operation"] == "remove_refined_from_relationship"
    assert change["from_name"] == _PREDECESSOR
    assert change["to_name"] == _SUCCESSOR
    assert change["reason"] == _REASON


def test_dry_run_reports_same_directed_intent_and_writes_nothing() -> None:
    from iter_codex.standard_names.edit import remove_refined_from_relationship

    graph = _LineageGraph()
    before = graph.shape()
    before_count = graph.relationship_count()

    result = remove_refined_from_relationship(
        _SUCCESSOR, _PREDECESSOR, reason=_REASON, gc=graph, dry_run=True
    )

    assert result["ok"] is True and result["dry_run"] is True
    assert result["direction"] == (
        "'beta' -[:REFINED_FROM]-> 'normalized_toroidal_plasma_beta'"
    )
    assert graph.relationship_count() == before_count
    assert graph.shape() == before
    assert graph.writes == [] and graph.changes == []


def test_refuses_nonexistent_endpoint_without_changing_relationship_count() -> None:
    from iter_codex.standard_names.edit import remove_refined_from_relationship

    graph = _LineageGraph()
    before_count = graph.relationship_count()

    result = remove_refined_from_relationship(
        "missing_successor", _PREDECESSOR, reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert "missing_successor" in result["reason"]
    assert "not found" in result["reason"]
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []


def test_refuses_reversed_direction_and_names_the_checked_direction() -> None:
    from iter_codex.standard_names.edit import remove_refined_from_relationship

    graph = _LineageGraph()
    before_count = graph.relationship_count()

    result = remove_refined_from_relationship(
        _PREDECESSOR, _SUCCESSOR, reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert (
        "'normalized_toroidal_plasma_beta' -[:REFINED_FROM]-> 'beta'"
        in result["reason"]
    )
    assert "reverse direction" in result["reason"]
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []


def test_refuses_to_strand_superseded_predecessor_and_keeps_edge() -> None:
    from iter_codex.standard_names.edit import remove_refined_from_relationship

    graph = _LineageGraph(alternate_successor=False)
    before_count = graph.relationship_count()

    result = remove_refined_from_relationship(
        _SUCCESSOR, _PREDECESSOR, reason=_REASON, gc=graph
    )

    assert result["ok"] is False
    assert "would strand superseded predecessor" in result["reason"]
    assert "0 remaining incoming successor edges" in result["reason"]
    assert graph.relationship_count() == before_count
    assert (_SUCCESSOR, _PREDECESSOR) in graph.edges
    assert graph.writes == [] and graph.changes == []


def test_allows_last_successor_edge_removal_when_predecessor_is_live() -> None:
    from iter_codex.standard_names.edit import remove_refined_from_relationship

    graph = _LineageGraph(predecessor_stage="accepted", alternate_successor=False)
    before_count = graph.relationship_count()

    result = remove_refined_from_relationship(
        _SUCCESSOR, _PREDECESSOR, reason=_REASON, gc=graph
    )

    assert result["ok"] is True
    assert graph.relationship_count() == before_count - 1
    assert (_SUCCESSOR, _PREDECESSOR) not in graph.edges
    assert len(graph.changes) == 1


def test_cli_exposes_required_reason_and_dry_run() -> None:
    from iter_codex.cli.sn import sn
    from iter_codex.standard_names.edit import remove_refined_from_relationship

    graph = _LineageGraph()
    before_count = graph.relationship_count()

    def _remove(successor: str, predecessor: str, **kwargs: Any) -> dict[str, Any]:
        return remove_refined_from_relationship(
            successor, predecessor, gc=graph, **kwargs
        )

    with patch(
        "iter_codex.standard_names.edit.remove_refined_from_relationship",
        side_effect=_remove,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "remove-lineage",
                _SUCCESSOR,
                _PREDECESSOR,
                "--reason",
                _REASON,
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "would remove 'beta' -[:REFINED_FROM]->" in result.output
    assert graph.relationship_count() == before_count
    assert graph.writes == [] and graph.changes == []

    missing_reason = CliRunner().invoke(
        sn, ["remove-lineage", _SUCCESSOR, _PREDECESSOR]
    )
    assert missing_reason.exit_code != 0
    assert "--reason" in missing_reason.output
