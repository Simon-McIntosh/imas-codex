"""Unit-authority contracts for derived parents and reviewed repairs."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_derived_parent_identity_rejects_bare_state_without_subject() -> None:
    from imas_codex.standard_names.graph_ops import (
        _validate_derived_parent_identity,
    )

    issues = _validate_derived_parent_identity(
        "internal_state_density",
        kind="scalar",
        unit="m^-3",
    )

    assert issues
    assert "requires a species subject" in issues[0]


def test_reviewed_unit_repair_is_ledgered_and_updates_edge() -> None:
    from imas_codex.standard_names.graph_ops import stamp_standard_name_units

    gc = MagicMock()
    gc.query.side_effect = [
        [{"id": "particle_energy"}, {"id": "hydrogenic_density"}],
        [{"id": "hydrogenic_density"}, {"id": "particle_energy"}],
    ]

    changed = stamp_standard_name_units(
        gc,
        {
            "particle_energy": "J",
            "hydrogenic_density": "m^-3",
        },
        reason="reviewed dimensional units for structural parents",
    )

    assert changed == ["hydrogenic_density", "particle_energy"]
    write_query = gc.query.call_args_list[1].args[0]
    assert "CREATE (change:StandardNameChange" in write_query
    assert "operation: 'repair_unit_authority'" in write_query
    assert "SET sn.unit = row.unit" in write_query
    assert "MERGE (sn)-[:HAS_UNIT]->(unit)" in write_query


def test_reviewed_unit_repair_refuses_missing_target_before_write() -> None:
    from imas_codex.standard_names.graph_ops import stamp_standard_name_units

    gc = MagicMock()
    gc.query.return_value = [{"id": "particle_energy"}]

    with pytest.raises(ValueError, match="hydrogenic_density"):
        stamp_standard_name_units(
            gc,
            {
                "particle_energy": "J",
                "hydrogenic_density": "m^-3",
            },
            reason="reviewed dimensional units for structural parents",
        )

    assert gc.query.call_count == 1


def test_reviewed_unit_repair_requires_reason() -> None:
    from imas_codex.standard_names.graph_ops import stamp_standard_name_units

    with pytest.raises(ValueError, match="reason"):
        stamp_standard_name_units(
            MagicMock(),
            {"particle_energy": "J"},
            reason=" ",
        )
