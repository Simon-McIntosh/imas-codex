"""Safety contracts for retiring unrecoverable provenance orphans."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_retirement_requires_accepted_opt_in() -> None:
    from imas_codex.standard_names.provenance_lifecycle import (
        retire_unrecoverable_provenance_orphans,
    )

    gc = MagicMock()
    gc.query.return_value = [{"id": "accepted_name", "stage": "accepted"}]

    with pytest.raises(ValueError, match="include_accepted"):
        retire_unrecoverable_provenance_orphans(gc, ["accepted_name"])

    assert gc.query.call_count == 1


def test_retirement_is_list_scoped_and_ledgered_atomically() -> None:
    from imas_codex.standard_names.provenance_lifecycle import (
        retire_unrecoverable_provenance_orphans,
    )

    gc = MagicMock()
    gc.query.side_effect = [
        [{"id": "accepted_name", "stage": "accepted"}],
        [{"id": "accepted_name"}],
    ]

    retired = retire_unrecoverable_provenance_orphans(
        gc,
        ["accepted_name"],
        include_accepted=True,
    )

    assert retired == ["accepted_name"]
    write = gc.query.call_args_list[1]
    cypher = write.args[0]
    assert "MATCH (sn:StandardName {id: $name_id})" in cypher
    assert "CREATE (change:StandardNameChange" in cypher
    assert "DETACH DELETE sn" in cypher
    assert write.kwargs["deletion_operation"] == "remove_provenance_orphan"


def test_parent_source_reconcile_refuses_inadmissible_scaffold() -> None:
    """Provenance recovery must not legitimize an invalid structural parent."""
    from imas_codex.standard_names.graph_ops import (
        reconcile_orphan_parent_sources,
    )
    from imas_codex.standard_names.parents import AdmissionResult

    gc = MagicMock()
    with (
        patch(
            "imas_codex.standard_names.graph_ops.find_orphan_parent_source_candidates",
            return_value=[
                {
                    "parent_id": "line_integrated_impurity_ion_velocity",
                    "origin": "derived",
                    "dd_paths": [
                        "spectrometer_x_ray_crystal/channel/"
                        "profiles_line_integrated/velocity_tor"
                    ],
                }
            ],
        ),
        patch(
            "imas_codex.standard_names.parents.is_admissible_parent_name",
            return_value=AdmissionResult(
                admit=False,
                reason="suppressed: single-child shadow",
                clause=None,
            ),
        ),
    ):
        seeded = reconcile_orphan_parent_sources(gc=gc)

    assert seeded == 0
    gc.query.assert_not_called()
