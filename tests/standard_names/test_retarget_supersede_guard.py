"""A supersede whose successor denotes a different DD path must be refused."""

from __future__ import annotations

from iter_codex.graph.models import RepairMutationKind
from iter_codex.standard_names.signed_manifest import (
    _LoadedRow,
    _retarget_supersede_refusal,
)

_PREDECESSOR = "separatrix"
_SUCCESSOR = "separatrix_average"


def _row(successor_id: str | None) -> _LoadedRow:
    arguments = {} if successor_id is None else {"successor_id": successor_id}
    return _LoadedRow(
        id=f"{_PREDECESSOR}=>{successor_id}",
        identity={"kind": "standard_name", "target_id": _PREDECESSOR},
        participants=(),
        mutations=(
            {
                "id": f"{_PREDECESSOR}:supersede",
                "order": 0,
                "kind": RepairMutationKind.supersede.value,
                "participant_id": _PREDECESSOR,
                "arguments": arguments,
            },
        ),
        guards=(),
        orphan_policy="refuse",
    )


def _snapshot(source_paths: list[str]) -> dict[str, object]:
    return {"properties": {"source_paths": source_paths}}


def _action(
    row: _LoadedRow, predecessor_paths: list[str], successor_paths: list[str]
) -> dict[str, object]:
    return {
        "row": row,
        "participant_snapshots": {
            _PREDECESSOR: _snapshot(predecessor_paths),
            _SUCCESSOR: _snapshot(successor_paths),
        },
    }


def test_retarget_onto_a_disjoint_dd_path_is_refused() -> None:
    row = _row(_SUCCESSOR)
    action = _action(
        row,
        predecessor_paths=["dd:equilibrium/boundary_separatrix/psi"],
        successor_paths=["dd:equilibrium/boundary_separatrix_average/psi"],
    )
    refusal = _retarget_supersede_refusal(None, action)
    assert refusal is not None
    assert _PREDECESSOR in refusal
    assert _SUCCESSOR in refusal


def test_supersede_sharing_a_bound_source_path_is_not_refused() -> None:
    row = _row(_SUCCESSOR)
    action = _action(
        row,
        predecessor_paths=["dd:equilibrium/boundary_separatrix/psi"],
        successor_paths=[
            "dd:equilibrium/boundary_separatrix/psi",
            "dd:equilibrium/boundary_separatrix_average/psi",
        ],
    )
    assert _retarget_supersede_refusal(None, action) is None


def test_predecessor_with_no_bound_source_has_nothing_to_retarget() -> None:
    row = _row(_SUCCESSOR)
    action = _action(row, predecessor_paths=[], successor_paths=["dd:some/other/path"])
    assert _retarget_supersede_refusal(None, action) is None


def test_unsigned_supersede_without_a_successor_is_left_to_other_guards() -> None:
    row = _row(None)
    action = _action(
        row,
        predecessor_paths=["dd:equilibrium/boundary_separatrix/psi"],
        successor_paths=[],
    )
    assert _retarget_supersede_refusal(None, action) is None


def test_non_supersede_mutation_is_ignored() -> None:
    row = _LoadedRow(
        id="unrelated",
        identity={},
        participants=(),
        mutations=(
            {
                "id": "unrelated:detach",
                "order": 0,
                "kind": RepairMutationKind.detach.value,
                "participant_id": _PREDECESSOR,
                "arguments": {},
            },
        ),
        guards=(),
        orphan_policy="refuse",
    )
    action = _action(row, predecessor_paths=["dd:x"], successor_paths=["dd:y"])
    assert _retarget_supersede_refusal(None, action) is None
