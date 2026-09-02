"""Publication and graph maintenance derive quantity kind from identity."""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.export import _graph_node_to_entry_dict
from imas_codex.standard_names.graph_ops import reconcile_standard_name_kinds

_STALE_TENSOR_NAMES = (
    "flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude",
    "plasma_electrical_conductivity",
)


@pytest.mark.parametrize("name", _STALE_TENSOR_NAMES)
def test_export_rederives_scalar_kind_from_canonical_name(name: str) -> None:
    entry = _graph_node_to_entry_dict(
        {
            "id": name,
            "kind": "tensor",
            "physics_domain": "equilibrium",
        }
    )

    assert entry["kind"] == "scalar"


class _KindGraph:
    """Minimal graph double retaining rows written by the kind reconciler."""

    def __init__(self) -> None:
        self.rows = {
            _STALE_TENSOR_NAMES[0]: "tensor",
            "plasma_current": "scalar",
        }
        self.changes: list[dict[str, Any]] = []
        self.statements: list[str] = []

    def query(self, statement: str, **parameters: Any) -> list[dict[str, Any]]:
        self.statements.append(statement)
        if "RETURN sn.id AS id, sn.kind AS stored_kind" in statement:
            return [
                {"id": name, "stored_kind": kind} for name, kind in self.rows.items()
            ]

        changed: list[dict[str, Any]] = []
        for row in parameters["updates"]:
            if self.rows.get(row["id"]) != row["stored_kind"]:
                continue
            self.rows[row["id"]] = row["derived_kind"]
            self.changes.append(row.copy())
            changed.append({"id": row["id"]})
        return changed

    def close(self) -> None:
        pass


def test_reconcile_refreshes_only_disagreement_with_change_ledger() -> None:
    graph = _KindGraph()

    result = reconcile_standard_name_kinds(gc=graph)

    stale_name = _STALE_TENSOR_NAMES[0]
    assert result == {"names_refreshed": 1}
    assert graph.rows == {stale_name: "scalar", "plasma_current": "scalar"}
    assert graph.changes == [
        {
            "id": stale_name,
            "stored_kind": "tensor",
            "derived_kind": "scalar",
        }
    ]
    write_statement = graph.statements[1]
    assert "CREATE (change:StandardNameChange" in write_statement
    assert "HAS_INTERNAL_CHANGE" in write_statement
    assert "SET sn.kind = row.derived_kind" in write_statement
